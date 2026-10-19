import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    from . import selective_scan_cuda_oflex as selective_scan_cuda
except ImportError:
    try:
        from . import selective_scan_cuda
    except ImportError:
        raise ImportError("找不到核心库，请检查 .so 文件")

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1: u = u.contiguous()
        if delta.stride(-1) != 1: delta = delta.contiguous()
        if D is not None: D = D.contiguous()
        if B.stride(-1) != 1: B = B.contiguous()
        if C.stride(-1) != 1: C = C.contiguous()
        if z is not None and z.stride(-1) != 1: z = z.contiguous()

        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()

        # Forward 之前验证通过了，说明 Forward 接口没问题，保持原样
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_softplus, 1, False)
        
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]
        
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x)
        return out if not return_last_state else (out, last_state)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, z, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1: dout = dout.contiguous()
        
        # 【修复点】删除了最后一个 False 参数
        # 错误日志显示 bwd 只需要 11 个参数，最后两个是 bool(delta_softplus) 和 int(nrows)
        du, ddelta, dA, dB, dC, dD, dz, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, dout, x, ctx.delta_softplus, 1
        )
        
        ddelta_bias = None
        if delta_bias is not None:
            ddelta_bias = ddelta.sum(dim=(0, 2))

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        
        return du, ddelta, dA, dB, dC, dD, dz, ddelta_bias, None, None

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
