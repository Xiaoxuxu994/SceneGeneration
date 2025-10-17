import torch
import torch_npu
from tqdm import trange
from kernel_custom_dist import kernel_custom_dist


@torch.inference_mode()
def distance(queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    Uniformly sample points on the mesh

    Args:
        queries: (M, 3) float Tensor
        targets: (N, 3) int Tensor
    Returns:
        distances: (M,) float Tensor
    '''
    assert queries.dim() == targets.dim() == 2
    assert queries.shape[1] == targets.shape[1] == 3
    assert queries.dtype == targets.dtype == torch.float
    CHUNK = 40960
    output = []
    x, y, z = (targets[:, i].contiguous() for i in range(3))
    for q in queries.split(CHUNK):
        buffer = q.new(q.shape[0])
        workspace = q.new(16 * CHUNK)
        kernel_custom_dist(
            q[:, 0].contiguous(),
            q[:, 1].contiguous(),
            q[:, 2].contiguous(),
            x, y, z, buffer, workspace
        )
        output.append(buffer)
    return torch.cat(output, 0)


@torch.inference_mode()
def sample_mesh(verts: torch.Tensor, faces: torch.Tensor, num_samples: int) -> torch.Tensor:
    '''
    Uniformly sample points on the mesh

    Args:
        verts: (V, 3) float Tensor
        faces: (F, 3) int Tensor
        num_samples: int
    Returns:
        points: (N,) float Tensor
    '''
    v = verts[faces]
    areas = (v[:, 0] - v[:, 1]).cross(v[:, 0] - v[:, 2], 1).norm(dim=1)
    areas = areas.cumsum(0)
    samples_per_face = (num_samples / areas[-1] * areas).round().int()
    samples_per_face[1:] = samples_per_face[1:] - samples_per_face[:-1]
    v = v.repeat_interleave(samples_per_face.clamp_(0), 0)
    samples = torch.rand(v.shape[0], 2, 1, dtype=v.dtype, device=v.device)
    samples = samples.where(samples.sum((1, 2), True) < 1, 1 - samples)
    return v[:, 0] + (samples * (v[:, 1:] - v[:, :1])).sum(1)


@torch.inference_mode()
def farthest_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    '''
    Evenly sample point cloud

    Args:
        points: (V, 3) float Tensor
        num_samples: int
    Returns:
        samples: (N, 3) float Tensor
    '''
    indices = [idx := 0]
    dists = (points - points[0]).norm(dim=1)
    for _ in trange(num_samples - 1):
        indices.append(idx := dists.argmax(0).item())
        dists = dists.minimum((points - points[idx]).norm(dim=1))
    return points[indices]