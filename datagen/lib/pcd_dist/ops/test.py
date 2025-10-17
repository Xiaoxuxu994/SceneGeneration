from time import perf_counter
import torch
from util import distance

DTYPE = torch.float
M, K, R = 102400, 1024, 10
a = torch.randn(M, 3, dtype=DTYPE, device=0)
b = torch.randn(K, 3, dtype=DTYPE, device=0)

d = torch.empty(M, dtype=DTYPE, device=0)
CHUNK = (1 << 30) // K
timer = perf_counter()
for _ in range(R):
    for i in range(0, M, CHUNK):
        d[i:i + CHUNK] = torch.cdist(a[i:i + CHUNK], b).amin(1)
d[-1].item()
print('torch.cdist time:', (perf_counter() - timer) / R, 'seconds')

timer = perf_counter()
for _ in range(R):
    udf = distance(a, b)
udf[-1].item()
print('custom op time:', (perf_counter() - timer) / R, 'seconds')
print('mean error:', (d - udf).abs().mean().item())
