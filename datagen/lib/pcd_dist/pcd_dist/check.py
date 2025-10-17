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
    print('Mean:', np.mean(diff), 'Max:', np.max(diff))


def unpack_s4tos8(x):
    x1 = (((x & 0xf) << 4) >> 4)
    x2 = (x & np.uint8(0xf0).view(np.int8)) >> 4
    x_unpack = np.stack([x1, x2], axis=-1).flatten()
    return x_unpack


if __name__ == '__main__':
    outmtx = np.fromfile("./workspace/output/output_outmtx.bin", dtype=np.float32)
    out_gt = np.fromfile("./workspace/input/output_outmtx.bin", dtype=np.float32)
    print('outmtx:', outmtx)
    print('out_gt:', out_gt)
    check_diff(outmtx, out_gt)
