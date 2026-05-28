import numpy as np
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, suppress=True)
from rich.text import Text
from rich.style import Style
from rich.console import Console
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))


def check_diff(x, y):
    diff = np.abs(x - y)
    arg = np.argmax(diff.flatten())
    print('Left:', x.flatten()[arg], 'Right:', y.flatten()[arg], 'Index:', arg)


def unpack_s4tos8(x):
    x1 = (((x & 0xf) << 4) >> 4)
    x2 = (x & np.uint8(0xf0).view(np.int8)) >> 4
    x_unpack = np.stack([x1, x2], axis=-1).flatten()
    return x_unpack


if __name__ == '__main__':
    M = 128*128*8
    K = 128*128*8
    
    xmtx = np.fromfile("./workspace/input/input_xmtx.bin", dtype=np.float32)
    ymtx = np.fromfile("./workspace/input/input_ymtx.bin", dtype=np.float32)
    zmtx = np.fromfile("./workspace/input/input_zmtx.bin", dtype=np.float32)
    x0mtx = np.fromfile("./workspace/input/input_x0mtx.bin", dtype=np.float32)
    y0mtx = np.fromfile("./workspace/input/input_y0mtx.bin", dtype=np.float32)
    z0mtx = np.fromfile("./workspace/input/input_z0mtx.bin", dtype=np.float32)
    outmtx = np.fromfile("./workspace/output/output_outmtx.bin", dtype=np.float32)
    
    workspace = np.fromfile('./workspace/output/output_workspace.bin', dtype=np.float16)
    
    # Computation and comparison
    p = np.stack((xmtx, ymtx, zmtx), -1)
    p0 = np.stack((x0mtx, y0mtx, z0mtx), -1)
    gt = np.sqrt(((p[:, None] - p0) ** 2).sum(-1).min(-1))
    check_diff(outmtx, gt)
