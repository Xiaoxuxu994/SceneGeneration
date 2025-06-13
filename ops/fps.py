import torch
from torch.autograd import Function

try:
    import fps_cpu
except ImportError:
    raise ImportError("Could not import fps_cpu. Please compile the extension.")

class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, points, lengths, K, start_idxs=None):
        if start_idxs is None:
            # 如果没有提供起始索引，默认选择每个点云的第一个点
            start_idxs = torch.zeros_like(lengths)
        
        # 调用C++实现
        return fps_cpu.farthest_point_sampling_cpu(points, lengths, K, start_idxs)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 此操作不可微，返回None梯度
        return None, None, None, None

def fps(points, lengths, K, start_idxs=None):
    """
    Python接口函数
    Args:
        points: (N, P, D) 点坐标
        lengths: (N,) 每个点云的实际点数
        K: (N,) 每个点云要采样的点数
        start_idxs: (N,) 可选，每个点云的起始采样索引
    Returns:
        (N, max_K) 采样点的索引
    """
    return FarthestPointSampling.apply(points, lengths, K, start_idxs)