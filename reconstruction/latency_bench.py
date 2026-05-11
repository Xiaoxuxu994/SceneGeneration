
import os
os.environ['ASCNED_RT_VISIBLE_DEVICES'] = '6'
os.environ['USE_CHECKPOINT'] = 'False'
import torch
from torch import nn
# from torchair import get_npu_backend
# from torchair.configs.compiler_config import CompilerConfig
# from torch_npu.npu import set_compile_mode
import torch_npu.npu
import torch_npu.profiler
from tqdm import trange
from submodules.pi3.models.pi3 import Pi3
from utils.config_loader import LoadConfig

class LinearW8A8Mxfp8(nn.Module):
    def __init__(self, float_module: nn.Linear, layer_name=None):
        super().__init__()
        # Offline mxfp8 quantization of weights
        ori_weight = float_module.weight.data.clone()
        if ori_weight.dtype == torch.float32:
            print(f'[Warning] force setting weight to fp16.')
            ori_weight = ori_weight.to(torch.float16)
        w_quant, w_mx_scale = torch_npu.npu_dynamic_mx_quant(ori_weight, dst_type=torch.float8_e4m3fn)
        # Use weightNZ: w_int8_nz = torch_npu.npu_format_cast(w_int8.t(), 29)
        self.register_buffer("w_quant", w_quant)  # [in, out]
        self.register_buffer("w_mx_scale", w_mx_scale)  # [out]
        self.bias = float_module.bias

        if layer_name:
            self.layer_name = layer_name
        else:
            self.layer_name = ""

    def forward(self, x):
        # Online mxfp8 quantization of activations
        ori_shape = x.shape
        ori_type = x.dtype
        if ori_type == torch.float32:
            x = x.to(torch.float16)
        if len(ori_shape) == 3:
            B, N, C = ori_shape
            x = x.view(-1, C)
        x_quant, x_mx_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        out = torch_npu.npu_quant_matmul(x_quant, self.w_quant.transpose(-1,-2),
                                    self.w_mx_scale.transpose(0, 1),
                                    pertoken_scale=x_mx_scale,
                                    bias=self.bias if self.bias is None else self.bias.float(),
                                    group_sizes=[1,1,32],
                                    # x1_dtype=torch.float8_e4m3fn, x2_dtype=torch.float8_e4m3fn,
                                    pertoken_scale_dtype=torch_npu.float8_e8m0fnu, scale_dtype=torch_npu.float8_e8m0fnu,
                                    output_dtype=x.dtype)
        if len(ori_shape) == 3:
            out = out.view(B, N, -1)
        return out.to(ori_type)

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def quant_model(model, quant_type="mxfp8"):
    cls_map = {"mxfp8": LinearW8A8Mxfp8}
    if quant_type not in cls_map:
        raise ValueError(f"Unknown quant_type={quant_type!r}, expected one of {list(cls_map)}")
    QCls = cls_map[quant_type]
    layer_counter = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _set_module(model, name, QCls(float_module=module, layer_name=name))
            layer_counter += 1
    print(f"[INFO] Replaced Quant Layer Num: {layer_counter}")

CONFIG = '/efs-m3ga/y00959808/SceneGeneration/reconstruction/configs/reliev3r_quant.json'
H, W = 1078, 1918
N_VIEWS = 12
DTYPE = 'mxfp8'
COMPILE = False

pi3 = Pi3(LoadConfig(CONFIG).model.depth_aligner.pi3)
pi3 = pi3.to('npu', torch.bfloat16).eval().requires_grad_(False)
if DTYPE == 'mxfp8': quant_model(pi3)
xpos_max = max(H//pi3.patch_size, W//pi3.patch_size)
images = torch.randn(N_VIEWS, 3, H, W, device='npu', dtype=torch.bfloat16)

def infer(images: torch.Tensor):
    hidden = pi3.encoder.forward(images, is_training=True)["x_norm_patchtokens"]
    return pi3.decode(hidden, N_VIEWS, H, W, xpos_max=xpos_max)[0]

for SPARSE in 2048, 4096, 8192, 16384, 32768, 65536:
    OUTPUT = f'/efs-m3ga/y00959808/SceneGeneration/benchmark/profiler/{DTYPE}{"-compile" if COMPILE else ""}-{N_VIEWS}x{H}x{W}-sparse{SPARSE}'

    if SPARSE > 1:
        patch_counter = 0
        for name, module in pi3.decoder.named_modules():
            if name.endswith('attn') and int(name.split('.')[0]) % 2:
                module.sparsity = SPARSE
                patch_counter += 1
        print(f"[INFO] Patched Sparse-Attention Module Num: {patch_counter}, with sparsity {SPARSE}")

    if COMPILE:
        # set_compile_mode(jit_compile=True)
        # npu_backend = get_npu_backend(compiler_config=CompilerConfig())
        # infer = torch.compile(infer, backend=npu_backend, dynamic=False)
        infer = torch.compile(infer, backend='npugraph_ex')

    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(0, 1, 1, 1, 0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(OUTPUT),
        record_shapes=True,
        # profile_memory=True,
        # with_stack=True,
        # with_flops=True,
        # with_modules=True,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            # aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            # l2_cache=True, msprof_tx=True, mstx=True,
            export_type=[torch_npu.profiler.ExportType.Text],
            # sys_io=True, sys_interconnection=True
        )
    )

    profiler.start()
    for _ in trange(2, desc=f'Sparse {SPARSE}'):
        infer(images)
        torch_npu.npu.synchronize()
        profiler.step()
    profiler.stop()