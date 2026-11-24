"""Forward-only efficiency measurement used for the public dataset."""
import contextlib
import inspect
import sys
import time
import traceback


def configure_flops(analysis):
    """Shared arithmetic exclusions and a real MAC handler for same-padding conv."""
    from fvcore.nn.jit_handles import get_shape, conv_flop_count
    def convolution_mode(inputs, outputs):
        # aten::_convolution_mode has no transposed flag; this is ordinary conv.
        shapes = [get_shape(inputs[0]), get_shape(inputs[1]), get_shape(outputs[0])]
        assert all(s is not None for s in shapes), 'Unknown convolution shapes'
        return {'conv': conv_flop_count(*shapes, transposed=False)}
    analysis.set_op_handle('aten::_convolution_mode', convolution_mode)
    # These have the same roles as the elementwise activations, arithmetic,
    # comparisons and memory-only operations already excluded for ALL models.
    for op in ['lift_fresh', 'leaky_relu', 'leaky_relu_', 'reciprocal',
               'sin', 'cos', 'dropout_', 'ones_like', 'all', 'rsub', 'expand_as']:
        analysis.set_op_handle('aten::'+op, None)
    # PythonOps are matched by their qualified Python class name. None is
    # insufficient in this fvcore version because ignore checks use node.kind().
    # CrossScan is memory movement; CrossMerge additionally has elementwise adds,
    # excluded by the same all-model arithmetic scope as aten::add above.
    for name in ('CrossScan', 'CrossMerge', 'CrossScanTriton', 'CrossMergeTriton',
                 'CrossScanF', 'CrossMergeF'):
        analysis.set_op_handle('prim::PythonOp.'+name, lambda inputs, outputs: 0)
    return analysis


@contextlib.contextmanager
def trace_compatibility():
    """Fixed-shape profiling only: keep real inference/timing code unchanged."""
    import torch
    originals = []
    def static_channel_mask(score, keep_p, sharp):
        if torch.isnan(score).any() or torch.isinf(score).any():
            score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
        channels = int(score.shape[1])
        k = max(1, min(int(round(float(keep_p)*channels)), channels))
        top, _ = torch.topk(score, k=k, dim=1)
        return torch.sigmoid(sharp*(score-top[:, -1].unsqueeze(1)))
    for name, module in list(sys.modules.items()):
        if name.endswith('.nn.modules.rram') and hasattr(module, 'soft_top_p_mask'):
            original = module.soft_top_p_mask
            source = inspect.getsource(original)
            if 'round(keep_p * C)' not in source:
                raise RuntimeError('Unknown PG mask implementation; refusing an unverified trace substitution')
            originals.append((module, original))
            module.soft_top_p_mask = static_channel_mask
    try:
        yield
    finally:
        for module, original in originals:
            module.soft_top_p_mask = original


def benchmark(model, settings, flops_only=False):
    import torch
    import statistics
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape

    class ForwardOnly(torch.nn.Module):
        def __init__(self, net):
            super().__init__(); self.net = net
        def forward(self, x):
            value = self.net(x)
            leaves = []
            def collect(obj):
                if isinstance(obj, torch.Tensor): leaves.append(obj)
                elif isinstance(obj, dict):
                    for key in sorted(obj): collect(obj[key])
                elif isinstance(obj, (tuple,list)):
                    for part in obj: collect(part)
            collect(value)
            return tuple(leaves)

    net = ForwardOnly(model).float().eval()
    x = torch.zeros((1,3,settings['imgsz'],settings['imgsz']), device='cuda', dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    repeats = []
    if not flops_only:
        with torch.inference_mode():
            for _ in range(settings['warmup']): net(x)
            torch.cuda.synchronize()
            for _ in range(settings['repeats']):
                t = time.perf_counter()
                for __ in range(settings['iterations']): net(x)
                torch.cuda.synchronize()
                repeats.append(settings['iterations']/(time.perf_counter()-t))
    else:
        # One untimed forward initializes cached shapes/anchors, just as the
        # original100 warmups did. This is executed ONLY by the user's launch.
        with torch.inference_mode():
            net(x)
            torch.cuda.synchronize()
    result = {'Params': sum(p.numel() for p in model.parameters()),
              'Params_M': sum(p.numel() for p in model.parameters())/1e6,
              'FLOPs_G': None, 'GPU': torch.cuda.get_device_name(),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'cudnn_version': torch.backends.cudnn.version(),
              'imgsz': settings['imgsz'], 'batch': 1, 'precision': 'FP32',
              'TF32': False, 'fusion': False, 'scope': settings['fps_scope']}
    if repeats:
        result.update(FPS=statistics.median(repeats), FPS_repeats=repeats,
                      latency_ms=1000/statistics.median(repeats))
    def scan(inputs, outputs):
        b,d,l = get_shape(inputs[0]); n = get_shape(inputs[2])[1]
        has_d = len(inputs)>5 and get_shape(inputs[5]) is not None
        # Official VMamba prefix-sum estimate in fvcore units; outer conversion x2.
        return 9*b*l*d*n + (b*d*l if has_d else 0)
    def scan_z(inputs, outputs):
        count = scan(inputs, outputs)
        if len(inputs)>6 and get_shape(inputs[6]) is not None:
            count += int(__import__('math').prod(get_shape(inputs[0])))
        return count
    def attention(inputs, outputs):
        q, k, v = [get_shape(i) for i in inputs[:3]]
        batch_heads = int(__import__('math').prod(q[:-2]))
        return batch_heads*q[-2]*k[-2]*(q[-1]+v[-1])
    def native_mha(inputs, outputs):
        q,k,v = [get_shape(i) for i in inputs[:3]]
        assert len(q)==len(k)==len(v)==3 and q[0]==k[0]==v[0]
        b,l,e=q; s=k[1]
        return b*(l+k[1]+v[1]+l)*e*e + 2*b*l*s*e
    try:
        analysis = FlopCountAnalysis(net, (x,)).tracer_warnings('none').unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        for name in ('SelectiveScanCore','SelectiveScanOflex','SelectiveScanCuda'):
            analysis.set_op_handle('prim::PythonOp.'+name, scan)
        analysis.set_op_handle('prim::PythonOp.SelectiveScanFn', scan_z)
        analysis.set_op_handle('aten::scaled_dot_product_attention', attention)
        analysis.set_op_handle('aten::_native_multi_head_attention', native_mha)
        # Same exclusions for every model: elementwise activations/arithmetic,
        # indexing, comparisons and memory rearrangement. Matrix products counted.
        ignored = ['add','add_','sub','sub_','mul','mul_','div','div_','pow','sqrt','rsqrt',
                   'sigmoid','sigmoid_','silu','silu_','gelu','relu','relu_','softmax',
                   'hardsigmoid','hardsigmoid_','hardswish','hardswish_','hardtanh','hardtanh_',
                   'softplus','exp','log','neg','sum','mean','amax','max','min','clamp',
                   'clamp_','clamp_min','floor','ceil','round','abs','sign','topk','sort',
                   'argsort','argmax','gather','scatter','scatter_','scatter_add_',
                   'index','index_put_','index_select','repeat','repeat_interleave',
                   'flip','roll','meshgrid','unfold','pad','constant_pad_nd',
                   'eq','ne','gt','lt','ge','le','logical_and','logical_or','bitwise_and',
                   'isfinite','remainder','where','dropout','feature_dropout','cumsum',
                   'isnan','isinf','any','nan_to_num','diag','diag_embed','diagonal',
                   'max_pool1d','max_pool2d','max_pool3d',
                   'adaptive_max_pool1d','adaptive_max_pool2d','adaptive_max_pool3d']
        for op in ignored:
            analysis.set_op_handle('aten::'+op, None)
        configure_flops(analysis)
        with trace_compatibility(), torch.inference_mode():
            units = float(analysis.total())
            unsupported = dict(analysis.unsupported_ops())
        result.update(FLOPs_estimate_G=2*units/1e9, unsupported_ops=unsupported,
                      FLOPs_method='2*fvcore FMA units; official selective-scan approximation; common elementwise/indexing/comparison exclusions (including max-pool comparisons); fixed-channel PG trace compatibility; same-padding convolution MAC handler; qualified PythonOp memory/elementwise handlers v4',
                      FLOPs_script_version=4,
                      FLOPs_status='incomplete' if unsupported else 'estimated_supported')
        if not unsupported:
            result['FLOPs_G'] = 2*units/1e9
        else:
            print('FLOPs incomplete, kept NA rather than silently omitting: '+str(unsupported), flush=True)
    except Exception as error:
        result['FLOPs_status'] = 'trace_failed'
        result['FLOPs_error'] = repr(error)
        result['FLOPs_traceback'] = traceback.format_exc()
        print('FLOPs trace failed, FPS/Params retained: '+repr(error), flush=True)
    return result
