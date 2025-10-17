import os
import numpy as np
from time import perf_counter
import torch

np.random.seed(43)


def pack_s8tos4(x):
    assert x.dtype==np.int8, "Must be type int8"
    x = x.reshape([-1, 2])
    x1 = x[:,0]
    x2 = x[:,1]
    x_pack = (x1 & 0xf) + (x2 << 4)
    return x_pack


def gen_golden_data_simple():
    M = 40960
    K = 1048576
    CHUNK = (1 << 30) // K

    xyz = torch.randn(M, 3, dtype=torch.float, device=0)
    xyz0 = torch.randn(K, 3, dtype=torch.float, device=0)
    udf = torch.empty(M, dtype=torch.float, device=0)
    timer = perf_counter()
    for _ in range(10):
        for i in range(0, M, CHUNK):
            udf[i:i + CHUNK] = torch.cdist(xyz[i:i + CHUNK], xyz0).amin(1)
    udf = udf.cpu().numpy()
    print('torch.cdist finished in', perf_counter() - timer, 'seconds.')
    xyz, xyz0 = xyz.cpu().numpy(), xyz0.cpu().numpy()

    xyz[:, 0].tofile('./workspace/input/input_xmtx.bin')
    xyz[:, 1].tofile('./workspace/input/input_ymtx.bin')
    xyz[:, 2].tofile('./workspace/input/input_zmtx.bin')
    xyz0[:, 0].tofile('./workspace/input/input_x0mtx.bin')
    xyz0[:, 1].tofile('./workspace/input/input_y0mtx.bin')
    xyz0[:, 2].tofile('./workspace/input/input_z0mtx.bin')
    udf.tofile('./workspace/input/output_outmtx.bin')


if __name__=="__main__":
    os.makedirs('./workspace/input/', exist_ok=True)
    os.makedirs('./workspace/output/', exist_ok=True)
    gen_golden_data_simple()
