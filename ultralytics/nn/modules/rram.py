from encodings.punycode import T
import math
import pickle
import numpy as np
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# ========= 开关与超参 =========
EN_A2_BLEND = False     # [A2] 关
A2_TOPK     = 5
A2_LAMBDA   = 0.30
A2_Tc       = 1.50
A2_Tg       = 1.00

EN_B1_CTX   = False     # [B1] 关
CTX_P3_ONLY = True

# [注意] 这里帮你改为 True，对应 train50 配置
EN_C1_SELKEEP = True    # [C1-out] 开：输出侧选择性保留
C1_KEEP_P     = 0.25    # train50 默认参数
C1_SHARP      = 10.0
C1_ETA        = 0.50

EN_C2_BOTTLENECK = False  # [C2] 关
C2_EPS           = 0.25

# ============================ 基础模块 ============================
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, H, A):
        H = H.to(self.weight.device)
        support = torch.matmul(H.to(torch.float32), self.weight.to(torch.float32))
        out = torch.matmul(A, support.to(A.dtype))
        return out + self.bias if self.bias is not None else out

def gen_A(num_classes, t, adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums'][:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    return torch.matmul(torch.matmul(A.to(D.dtype), D).t(), D)

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), True)
        s = self.fc2(s)
        s = F.hardsigmoid(s, True)
        return s * x

class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = None, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        if k_size is None:
            k_size = int(abs((math.log2(channels) / gamma) + b))
            k_size = k_size if k_size % 2 else k_size + 1
            k_size = max(3, k_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x).squeeze(-1).transpose(1, 2)
        y = self.conv1d(y)
        y = self.act(y).transpose(1, 2).unsqueeze(-1)
        return x * y

def eca_ksize(C: int, gamma: float = 2.0, b: float = 1.0) -> int:
    k = int(abs((math.log2(max(C, 1)) / gamma) + b)); k = k if k % 2 else k + 1
    return max(3, k)

class CalibratedSigmoid(nn.Module):
    def __init__(self, T: float = 1.15, beta: float = 0.02):
        super().__init__(); self.T, self.beta = T, beta
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aw = torch.sigmoid(x / self.T)
        return aw * (1.0 - self.beta) + self.beta if self.beta > 0 else aw

# ============================ [B1] Context‑Adapter ============================
class ContextAdapter(nn.Module):
    """DW 3x3(d=1/3/5) 并联 → 1x1 合并"""
    def __init__(self, C: int):
        super().__init__()
        self.dw1 = nn.Conv2d(C, C, 3, padding=1, dilation=1, groups=C, bias=False)
        self.dw3 = nn.Conv2d(C, C, 3, padding=3, dilation=3, groups=C, bias=False)
        self.dw5 = nn.Conv2d(C, C, 3, padding=5, dilation=5, groups=C, bias=False)
        self.pw   = nn.Conv2d(3*C, C, 1, bias=False)
        self.bn   = nn.BatchNorm2d(C)
        self.act  = nn.SiLU()
    def forward(self, x):
        a = self.dw1(x); b = self.dw3(x); c = self.dw5(x)
        y = torch.cat([a, b, c], dim=1)
        y = self.pw(y); y = self.bn(y)
        return self.act(y)

# [修复] 增加 NaN 清洗和边界保护的 Soft Top-P Mask
def soft_top_p_mask(score: torch.Tensor, keep_p: float, sharp: float) -> torch.Tensor:
    import torch  # 如果文件顶部没导入 torch，请在这里加上
    # 1. 清洗 NaN/Inf，防止 topk 崩溃
    if torch.isnan(score).any() or torch.isinf(score).any():
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

    B, C = score.shape
    # 2. 严格限制 k 的范围在 [1, C] 之间
    k = int(round(keep_p * C))
    k = max(1, min(k, C))  # <--- 强制 Clamp，防止越界
    
    # 3. 计算阈值
    topk_vals, _ = torch.topk(score, k=k, dim=1)
    thr = topk_vals[:, -1].unsqueeze(1)
    
    # 4. 生成掩码
    return torch.sigmoid(sharp * (score - thr))

class BottleneckRestore(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.reduce = nn.Conv2d(C, C//2, 1, bias=False)
        self.expand = nn.Conv2d(C//2, C, 1, bias=False)
        self.proj   = nn.Conv2d(C, C//2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C//2)
        self.bn2 = nn.BatchNorm2d(C)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(C, C, 1, bias=False)
    def forward(self, x, prior_short=None, eps=0.25):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.act(self.bn2(self.expand(y)))
        if prior_short is not None:
            y = y + eps * self.skip(prior_short)
        y = self.proj(y)
        return y

# ============================ 主模块 ============================
class RRAM(nn.Module):
    def __init__(self,
                 c1, c2,
                 num_classes=5, t=0.4,
                # adj_file='/path/to/adj.pkl',           1(共1)
                 adj_file = 'adj.pkl',
                 inp_name='/path/to/glove_wordEmbedding.pkl',
                 a_path=None, v_path=None):
        super().__init__()
        # [train50 核心] 使用 GMP
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.gc1 = GraphConvolution(300, 1024)
        self.gc2 = None
        self.relu = nn.LeakyReLU(0.2)  # train50 原配激活
        self.act  = nn.SiLU()

        def is_npy(x): return isinstance(x, str) and x.endswith('.npy')
        def is_pkl(x): return isinstance(x, str) and x.endswith('.pkl')
        if a_path is None and is_npy(adj_file): a_path, adj_file = adj_file, None
        if v_path is None and is_npy(inp_name): v_path, inp_name = inp_name, None
        if a_path is None and is_npy(num_classes): a_path, num_classes = num_classes, 5
        if v_path is None and is_npy(t): v_path, t = t, 0.0

        if (a_path is not None) and (v_path is not None):
            _adj = np.load(a_path).astype(np.float32)
            self.A   = Parameter(torch.from_numpy(_adj).float())
            self.inp = np.load(v_path).astype(np.float32)
            self.num_classes = int(_adj.shape[0])
        else:
            if num_classes is None and is_pkl(adj_file):
                with open(adj_file, 'rb') as f:
                    tmp = pickle.load(f)
                    num_classes = int(tmp['adj'].shape[0])
            if not is_pkl(adj_file):
                raise ValueError("RRAM expects either (.npy a_path,v_path) or a valid .pkl adj_file/inp_name")
            _adj = gen_A(int(num_classes), t, adj_file)
            self.A = Parameter(torch.from_numpy(_adj).float())
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
            self.num_classes = int(num_classes)

        self.linear = nn.Linear(self.num_classes, 1024)
        self.linear64  = nn.Linear(1024, 64)
        self.linear128 = nn.Linear(1024, 128)
        self.linear256 = nn.Linear(1024, 256)
        self.linear512 = nn.Linear(1024, 512)

        self.gate    = CalibratedSigmoid(T=1.15, beta=0.02)
        self.gate_p3 = CalibratedSigmoid(T=1.15, beta=0.03)

        self.se_layer64  = ECALayer(64,  k_size=5)
        self.se_layer128 = ECALayer(128, eca_ksize(128))
        self.se_layer256 = ECALayer(256, eca_ksize(256))
        self.se_layer512 = ECALayer(512, eca_ksize(512))

        self.conv64  = nn.Conv2d(64,  32, 1)
        self.conv128 = nn.Conv2d(128, 64, 1)
        self.conv256 = nn.Conv2d(256, 128,1)
        self.conv512 = nn.Conv2d(512, 256,1)

        if EN_B1_CTX:
            self.ctx64  = ContextAdapter(64)
            self.ctx128 = ContextAdapter(128) if not CTX_P3_ONLY else None
            self.ctx256 = ContextAdapter(256) if not CTX_P3_ONLY else None
            self.ctx512 = ContextAdapter(512) if not CTX_P3_ONLY else None
        else:
            self.ctx64 = self.ctx128 = self.ctx256 = self.ctx512 = None

        if EN_C2_BOTTLENECK:
            self.c2_64  = BottleneckRestore(64)
            self.c2_128 = BottleneckRestore(128)
            self.c2_256 = BottleneckRestore(256)
            self.c2_512 = BottleneckRestore(512)
        else:
            self.c2_64 = self.c2_128 = self.c2_256 = self.c2_512 = None

        self._dyn_linear = {}
        self._dyn_se     = {}
        self._dyn_conv   = {}

    def _ensure_gc2(self, C, device):
        need = (self.gc2 is None) or (getattr(self.gc2, "out_features", None) != int(C))
        if need:
            self.gc2 = GraphConvolution(1024, int(C)).to(device)

    def _dyn_branch(self, C, device):
        C = int(C)
        lin = self._dyn_linear.get(C)
        se  = self._dyn_se.get(C)
        conv= self._dyn_conv.get(C)
        if lin is None:  lin  = self._dyn_linear.setdefault(C, nn.Linear(1024, C))
        if se  is None:  se   = self._dyn_se.setdefault(C,  ECALayer(C, eca_ksize(C)))
        if conv is None: conv = self._dyn_conv.setdefault(C, nn.Conv2d(C, max(C // 2, 1), 1))
        for m in (lin, se, conv):
            params = list(m.parameters())
            if params and params[0].device != device: m.to(device)
        return lin, se, conv

    def _maybe_ctx(self, C, x: torch.Tensor) -> torch.Tensor:
        if not EN_B1_CTX: return x
        if C == 64 and self.ctx64 is not None:  return self.ctx64(x)
        if C == 128 and self.ctx128 is not None: return self.ctx128(x)
        if C == 256 and self.ctx256 is not None: return self.ctx256(x)
        if C == 512 and self.ctx512 is not None: return self.ctx512(x)
        return x

    def _a2_blend(self, s_raw: torch.Tensor, P_nC: torch.Tensor, g_mlp: torch.Tensor) -> torch.Tensor:
        if not EN_A2_BLEND:
            return g_mlp
        if A2_TOPK and A2_TOPK > 0 and A2_TOPK < P_nC.shape[0]:
            topk_val, topk_idx = torch.topk(s_raw, k=A2_TOPK, dim=1)
            mask = torch.full_like(s_raw, float('-inf'))
            mask.scatter_(1, topk_idx, topk_val)
            logits = mask / A2_Tc
        else:
            logits = s_raw / A2_Tc
        alpha = torch.softmax(logits, dim=1)
        g_prior = torch.matmul(alpha, P_nC)
        g_prior = torch.sigmoid(g_prior / A2_Tg)
        g = A2_LAMBDA * g_prior + (1.0 - A2_LAMBDA) * g_mlp
        return g

    # [C1] 输出侧选择性保留 (train50 核心)
    def _c1_outkeep(self, y0: torch.Tensor, att: torch.Tensor, conv_reduce: nn.Conv2d) -> torch.Tensor:
        if not EN_C1_SELKEEP:
            return conv_reduce(y0)  
        
        # y0 这里是 feat (SE之后)，需要先降维
        y0_out = conv_reduce(y0)
        
        B, C_out, H, W = y0_out.shape
        w_in = att.flatten(1)
        # 获取降维卷积权重绝对值
        Wabs = conv_reduce.weight.detach().abs().view(C_out, -1)
        # 计算输出通道重要性
        g_out = torch.matmul(w_in.to(Wabs.dtype), Wabs.t())
        # 软掩码
        m_out = soft_top_p_mask(g_out, keep_p=C1_KEEP_P, sharp=C1_SHARP)
        gain = (1.0 - C1_ETA) + C1_ETA * m_out
        
        return y0_out * gain.unsqueeze(-1).unsqueeze(-1)

    def forward(self, F_in: torch.Tensor):
        B, C, H, W = F_in.shape

        F_ctx = self._maybe_ctx(C, F_in)

        # GMP 摘要
        f = self.pooling(F_ctx).view(B, -1)

        inp = torch.tensor(self.inp, device=F_in.device)
        A_hat = gen_adj(self.A).detach()
        H1 = self.gc1(inp, A_hat); H1 = self.relu(H1)
        self._ensure_gc2(C, F_in.device)
        P_nC = self.gc2(H1, A_hat)

        s_raw = torch.matmul(f.to(P_nC.dtype), P_nC.transpose(0,1))
        #x1024 = self.linear(s_raw)
        x1024 = self.linear(s_raw.to(self.linear.weight.dtype))

        if C == 64:
            g_pre = self.linear64(x1024); g_mlp = self.gate_p3(g_pre)
            g = self._a2_blend(s_raw, P_nC, g_mlp)
            att = g.unsqueeze(-1).unsqueeze(-1)
            feat = self.se_layer64(F_in * att)
            if EN_C2_BOTTLENECK:
                y0 = self.c2_64(feat, prior_short=feat, eps=C2_EPS)
            elif EN_C1_SELKEEP:
                y0 = self._c1_outkeep(feat, att, self.conv64)
            else:
                y0 = self.conv64(feat)
            base = self.conv64(F_in)
            return torch.cat([base, y0], dim=1)

        elif C == 128:
            g_pre = self.linear128(x1024); g_mlp = self.gate(g_pre)
            g = self._a2_blend(s_raw, P_nC, g_mlp)
            att = g.unsqueeze(-1).unsqueeze(-1)
            feat = self.se_layer128(F_in * att)
            if EN_C2_BOTTLENECK:
                y0 = self.c2_128(feat, prior_short=feat, eps=C2_EPS)
            elif EN_C1_SELKEEP:
                y0 = self._c1_outkeep(feat, att, self.conv128)
            else:
                y0 = self.conv128(feat)
            base = self.conv128(F_in)
            return torch.cat([base, y0], dim=1)

        elif C == 256:
            g_pre = self.linear256(x1024); g_mlp = self.gate(g_pre)
            g = self._a2_blend(s_raw, P_nC, g_mlp)
            att = g.unsqueeze(-1).unsqueeze(-1)
            feat = self.se_layer256(F_in * att)
            if EN_C2_BOTTLENECK:
                y0 = self.c2_256(feat, prior_short=feat, eps=C2_EPS)
            elif EN_C1_SELKEEP:
                y0 = self._c1_outkeep(feat, att, self.conv256)
            else:
                y0 = self.conv256(feat)
            base = self.conv256(F_in)
            return torch.cat([base, y0], dim=1)

        elif C == 512:
            g_pre = self.linear512(x1024); g_mlp = self.gate(g_pre)
            g = self._a2_blend(s_raw, P_nC, g_mlp)
            att = g.unsqueeze(-1).unsqueeze(-1)
            feat = self.se_layer512(F_in * att)
            if EN_C2_BOTTLENECK:
                y0 = self.c2_512(feat, prior_short=feat, eps=C2_EPS)
            elif EN_C1_SELKEEP:
                y0 = self._c1_outkeep(feat, att, self.conv512)
            else:
                y0 = self.conv512(feat)
            base = self.conv512(F_in)
            return torch.cat([base, y0], dim=1)

        else:
            lin, se, conv = self._dyn_branch(C, F_in.device)
            g_pre = lin(x1024); g_mlp = self.gate(g_pre)
            g = self._a2_blend(s_raw, P_nC, g_mlp) if EN_A2_BLEND else g_mlp
            att = g.unsqueeze(-1).unsqueeze(-1)
            feat = se(F_in * att)
            y0 = conv(feat)
            if EN_C1_SELKEEP: y0 = self._c1_outkeep(feat, att, conv)
            base = conv(F_in)
            return torch.cat([base, y0], dim=1)
