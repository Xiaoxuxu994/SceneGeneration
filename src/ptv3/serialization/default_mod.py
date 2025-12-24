import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_

def _ensure_cpu(x: torch.Tensor) -> torch.Tensor:
    # 明确固定在 CPU，避免 NPU 上 bitwise/argsort 回退导致的性能/显存波动
    return x if x.device.type == "cpu" else x.to("cpu", non_blocking=True)


def _ensure_int_coords(grid_coord: torch.Tensor) -> torch.Tensor:
    # grid_coord 只需要整数坐标；优先 int32（省内存），但最终参与 bitwise 时会转成 int64
    if grid_coord.dtype not in (torch.int32, torch.int64):
        grid_coord = grid_coord.to(torch.int32)
    return grid_coord

@torch.no_grad()
def encode(grid_coord: torch.Tensor, batch: torch.Tensor = None, depth: int = 16, order: str = "z"):
    """
    Encode (grid_coord, batch) -> int64 code on CPU.
    注意：建议只在 CPU 上调用（内部会强制搬 CPU）。
    """
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}

    grid_coord = _ensure_int_coords(grid_coord)
    grid_coord = _ensure_cpu(grid_coord)

    # 选择编码输入（避免额外拷贝：advanced indexing 会拷贝，但这里只是 3 列，代价很小）
    if order in ("z-trans", "hilbert-trans"):
        gc = grid_coord[:, [1, 0, 2]]
    else:
        gc = grid_coord

    if order.startswith("z"):
        code = z_order_encode(gc, depth=depth)            # int64 on CPU
    else:
        code = hilbert_encode(gc, depth=depth)            # int64 on CPU

    if batch is not None:
        batch = _ensure_cpu(batch)
        # 允许 int32/int64，尽量避免 batch.long() 每次都做
        if batch.dtype not in (torch.int32, torch.int64):
            batch = batch.to(torch.int32)
        # bitwise 统一在 int64 上做
        batch64 = batch.to(torch.int64)
        code = (batch64 << (depth * 3)) | code

    # 强制返回 CPU int64
    return code.to(torch.int64)


@torch.no_grad()
def decode(code: torch.Tensor, depth: int = 16, order: str = "z"):
    """
    Decode int64 code -> (grid_coord[int32], batch[int32]) on CPU.
    注意：建议只在 CPU 上调用（内部会强制搬 CPU）。
    """
    assert order in {"z", "hilbert"}

    code = _ensure_cpu(code).to(torch.int64)

    # batch / code 拆分：都在 CPU int64 完成
    batch = code >> (depth * 3)
    code_xyz = code & ((1 << (depth * 3)) - 1)

    if order == "z":
        grid_coord = z_order_decode(code_xyz, depth=depth)     # [N,3] int64 (CPU)
    else:
        grid_coord = hilbert_decode(code_xyz, depth=depth)     # [N,3] int64 (CPU)

    # 返回更省内存的 dtype（通常够用）
    return grid_coord.to(torch.int32), batch.to(torch.int32)


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    # grid_coord: CPU int32/int64
    grid_coord = _ensure_cpu(_ensure_int_coords(grid_coord))
    x = grid_coord[:, 0].to(torch.int64)
    y = grid_coord[:, 1].to(torch.int64)
    z = grid_coord[:, 2].to(torch.int64)
    # b=None：保持和你原逻辑一致
    return z_order_encode_(x, y, z, b=None, depth=depth).to(torch.int64)


def z_order_decode(code: torch.Tensor, depth: int):
    code = _ensure_cpu(code).to(torch.int64)
    x, y, z = z_order_decode_(code, depth=depth)  # usually int64 CPU
    return torch.stack([x, y, z], dim=-1)         # [N,3] int64 CPU


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    grid_coord = _ensure_cpu(_ensure_int_coords(grid_coord))
    # hilbert_encode_ 一般期望 int/long；这里保持 int32 即可
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth).to(torch.int64)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    code = _ensure_cpu(code).to(torch.int64)
    grid_coord = hilbert_decode_(code, num_dims=3, num_bits=depth)  # [N,3], usually int64
    return grid_coord