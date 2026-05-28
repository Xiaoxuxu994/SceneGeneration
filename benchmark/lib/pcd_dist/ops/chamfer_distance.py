from torch import Tensor
from kernel_pcd_dist import kernel_pcd_dist

def distance(src: Tensor, tgt: Tensor) -> Tensor:
    dist = src.new_zeros(src.shape[0])
    kernel_pcd_dist(src, tgt, dist, src.new_zeros(256))
    return dist

def chamfer_distance(x: Tensor, y: Tensor) -> Tensor:
    return distance(x, y).mean() + distance(y, x).mean()

def chamfer_distance_batch(x: Tensor, y: Tensor) -> Tensor:
    '''
    Chamfer Distance
    Args:
        x (Tensor: (..., B,) M, 3):
        y (Tensor: (..., B,) N, 3):
    Returns:
        out (Tensor: (..., B)):
    '''
    if x.ndim == 2: chamfer_distance(x, y); return chamfer_distance(x, y)
    x_, y_ = x.flatten(0, -3).float(), y.flatten(0, -3).float()
    d = x_.new_zeros(x_.shape[0])
    chamfer_distance(x_[0], y_[0])
    for i in range(x_.shape[0]): d[i] = chamfer_distance(x_[i], y_[i])
    return d.view(*x.shape[:-2])

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 518*518*1, 3, device=7)
    y = torch.randn(1, 518*518*1, 3, device=7)
    # d = torch.cdist(x, y)
    # d = d.amin(-2).mean(-1) + d.amin(-1).mean(-1)
    cd = chamfer_distance_batch(x, y)
    # diff = (d - cd).abs()
    # print('mean error:', diff.mean().item(), 'max error:', diff.max().item())
    print("chamfer dist NPU: {}".format(cd.item()))
