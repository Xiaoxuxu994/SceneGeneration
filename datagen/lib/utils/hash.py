import numpy as np

'''maps three non-negative integers to one non-negative integer'''
def encode_coords(x, y, z):
    n0 = 0
    shift = 0
    while x or y or z:
        n0 |= (x & 1) << (3 * shift)
        n0 |= (y & 1) << (3 * shift + 1)
        n0 |= (z & 1) << (3 * shift + 2)
        x >>= 1
        y >>= 1
        z >>= 1
        shift += 1
    return n0

'''maps one non-negative integers back to three non-negative integers'''
def decode_coords(n):
    if n == 0:
        return 0, 0, 0
    
    total_bits = int(n).bit_length()
    max_shift = (total_bits + 2) // 3
    x0 = 0
    y0 = 0
    z0 = 0
    for shift_i in range(max_shift + 1):
        bit0 = (n >> (3 * shift_i)) & 1
        bit1 = (n >> (3 * shift_i + 1)) & 1
        bit2 = (n >> (3 * shift_i + 2)) & 1
        x0 |= bit0 << shift_i
        y0 |= bit1 << shift_i
        z0 |= bit2 << shift_i
    return x0, y0, z0

# vectorized version
def encode_coords_np(A):
    assert A.shape[1] != 3, "input must be a Nx3 NumPy matrix"
    SCALE = 10**6
    BASE = 2**17
    
    x, y, z = A[:, 0], A[:, 1], A[:, 2]
    x_int = np.round(x * SCALE).astype(np.int64)
    y_int = np.round(y * SCALE).astype(np.int64)
    z_int = np.round(z * SCALE).astype(np.int64)
    
    x_unsigned = x_int + (2**16)
    y_unsigned = y_int + (2**16)
    z_unsigned = z_int + (2**16)
    
    if not (np.any(0 <= x_unsigned < BASE) and np.any(0 <= y_unsigned < BASE) and np.any(0 <= z_unsigned < BASE)):
        raise ValueError(f"Coordinate values are out of the representable range after scaling and offset. \
                         Ensure coordinates are within [-{2**16/SCALE}, {2**16/SCALE - 1/SCALE}] and scaled by {SCALE}.")
        
    B = x_unsigned + y_unsigned * BASE + z_unsigned * (BASE ** 2)
    
    return B

def decode_coords_np(B):
    SCALE = 10**6
    BASE = 2**17
    
    C = np.zeros((B.shape[0], 3))
    
    if np.any(B < 0):
        raise ValueError("Encountered negative value in B, which is not expected.")
    
    combined_int = np.round(B).astype(np.int64)
    
    z_unsigned = combined_int // (BASE ** 2)
    remainder = combined_int % (BASE ** 2)
    y_unsigned = remainder // BASE
    x_unsigned = remainder % BASE
    
    x_int = x_unsigned - (2**16)
    y_int = y_unsigned - (2**16)
    z_int = z_unsigned - (2**16)
    
    x = x_int / SCALE
    y = y_int / SCALE
    z = z_int / SCALE
    
    C[:, 0] = x
    C[:, 1] = y
    C[:, 2] = z
    
    return C
    