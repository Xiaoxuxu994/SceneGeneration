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