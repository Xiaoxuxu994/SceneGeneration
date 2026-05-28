import os
import numpy as np

np.random.seed(43)


def pack_s8tos4(x):
    assert x.dtype==np.int8, "Must be type int8"
    x = x.reshape([-1, 2])
    x1 = x[:,0]
    x2 = x[:,1]
    x_pack = (x1 & 0xf) + (x2 << 4)
    return x_pack


def gen_golden_data_simple():
    M = 128*128*8
    K = 128*128*8
    
    xmtx = np.random.random([M]).astype(np.float32) * 0.5
    ymtx = np.random.random([M]).astype(np.float32) * 0.5
    zmtx = np.random.random([M]).astype(np.float32) * 0.5
    x0mtx = np.random.random([K]).astype(np.float32) * 0.5
    y0mtx = np.random.random([K]).astype(np.float32) * 0.5
    z0mtx = np.random.random([K]).astype(np.float32) * 0.5
    outmtx = np.random.random([M]).astype(np.float32) * 0.5
    
    # # In case ones are needed 
    # xmtx = np.ones([M]).astype(np.float32)
    # ymtx = np.ones([M]).astype(np.float32)
    # zmtx = np.ones([M]).astype(np.float32)
    # x0mtx = np.ones([K]).astype(np.float32)
    # y0mtx = np.ones([K]).astype(np.float32)
    # z0mtx = np.ones([K]).astype(np.float32)
    # outmtx = np.ones([M]).astype(np.float32)
    
    
    xmtx.tofile('./workspace/input/input_xmtx.bin')
    ymtx.tofile('./workspace/input/input_ymtx.bin')
    zmtx.tofile('./workspace/input/input_zmtx.bin')
    x0mtx.tofile('./workspace/input/input_x0mtx.bin')
    y0mtx.tofile('./workspace/input/input_y0mtx.bin')
    z0mtx.tofile('./workspace/input/input_z0mtx.bin')
    outmtx.tofile('./workspace/output/output_outmtx.bin')


if __name__=="__main__":
    os.makedirs('./workspace/input/', exist_ok=True)
    os.makedirs('./workspace/output/', exist_ok=True)
    gen_golden_data_simple()
    
