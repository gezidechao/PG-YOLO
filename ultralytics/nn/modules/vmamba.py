import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ultralytics.nn.modules.selective_scan_pkg import selective_scan_fn

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias.requires_grad = False
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = S4D_A_init(d_state, d_inner, copies=copies, device=device, merge=merge)
        A_log = torch.log(A) 
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner if not merge else d_inner * copies, device=device)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        # CPU 旁路：如果输入不是 CUDA Tensor，直接返回，跳过 CUDA 算子
        if not x.is_cuda:
            return x
            
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H) + inv_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return (out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y).view(B, -1, H, W)

    def forward(self, x: torch.Tensor):
        x_in = x
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        
        # 【关键修复】无论 CPU 还是 GPU，forward_core 返回的都是 NCHW
        # 所以必须转回 NHWC 才能和 z (NHWC) 相乘
        y = y.permute(0, 2, 3, 1).contiguous()
            
        y = y * F.silu(z)
        #out = self.out_norm(y)
        out = self.out_norm(y.to(self.out_norm.weight.dtype))
        out = self.dropout(out)
        return self.out_proj(out)
    


def S4D_A_init(d_state, d_inner, copies=1, device=None, merge=True):
    A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(d_inner, 1)
    if copies > 1:
        A = A.repeat(copies, 1)
    if merge:
        A = A.flatten(0, 1)
    A = 0.5 * math.pi * A
    return A

class VSSBlock(nn.Module):
    def __init__(self, c1, c2, drop_path=0., **kwargs):
        super().__init__()
        dim = c2
        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = SS2D(d_model=dim, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # 🔥 【核心改进】：增加局部纹理保留分支 (Local Texture Bypass)
        # 专门针对 Rebar (细长) 和 Corrosion (斑块) 等极易在1D扫描中丢失的细粒度共现特征
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)
        self.local_act = nn.SiLU()

    def forward(self, input: torch.Tensor):
        # 1. 提取局部细粒度特征 (保持 NCHW 格式，完美保留 2D 空间结构)
        local_feat = self.local_act(self.local_bn(self.local_conv(input)))

        # 2. Mamba 全局共现扫描 (转换为 NHWC 格式进行 1D 序列扫描)
        x = input.permute(0, 2, 3, 1).contiguous()
        x_norm = self.ln_1(x.to(self.ln_1.weight.dtype))
        global_feat = self.drop_path(self.self_attention(x_norm))

        # 3. 局部 (Rebar细节) 与 全局 (Spall上下文) 深度融合
        x = x + global_feat
        x = x.permute(0, 3, 1, 2).contiguous() # 转回 NCHW
        x = x + local_feat # 🔥 将未被破坏的局部高频特征注入回主流

        # 4. MLP 通道混合
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_nhwc = x_nhwc + self.drop_path(self.mlp(self.ln_2(x_nhwc.to(self.ln_2.weight.dtype))))

        # 5. 返回 NCHW 格式
        return x_nhwc.permute(0, 3, 1, 2).contiguous()
