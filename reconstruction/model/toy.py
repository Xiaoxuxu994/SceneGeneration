import torch
from torch import nn, Tensor


def svd_orthogonalize(m: Tensor):
    """Convert 9D representation to SO(3) using SVD orthogonalization.
    Args:
        m: [BATCH, 3, 3] 3x3 matrices.
    Returns:
        [BATCH, 3, 3] SO(3) rotation matrices.
    """
    mT = torch.nn.functional.normalize(m, dim=-1).mT
    u, _, v = torch.svd(mT)
    def det_3x3(mat: Tensor):
        return (mat[..., 0, 0] * mat[..., 1, 1] * mat[..., 2, 2] + 
                mat[..., 0, 1] * mat[..., 1, 2] * mat[..., 2, 0] + 
                mat[..., 0, 2] * mat[..., 1, 0] * mat[..., 2, 1] - 
                mat[..., 0, 2] * mat[..., 1, 1] * mat[..., 2, 0] - 
                mat[..., 0, 0] * mat[..., 1, 2] * mat[..., 2, 1] - 
                mat[..., 0, 1] * mat[..., 1, 0] * mat[..., 2, 2])
    det = det_3x3(v @ u.mT)
    # Check orientation reflection.
    r = torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2) @ u.mT
    return r


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xy = nn.Parameter(torch.randn(32, 2, 288, 512))
        self.z = nn.Parameter(torch.randn(32, 1, 288, 512))
        self.c2w_r = nn.Parameter(torch.randn(32, 3, 3))
        self.c2w_t = nn.Parameter(torch.randn(32, 3, 1))
        self.w2c_0001 = nn.Buffer(torch.tensor((0., 0., 0., 1.)))

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        view_ids = data['view_ids'][0]
        depth = self.z[None, view_ids].exp()
        # rays = torch.cat((self.xy[None, view_ids], torch.ones_like(depth)), 2)
        intr = data['input_Ks']
        B, V, _, H, W = depth.shape
        x = torch.linspace(.5 / W, 1 - .5 / W, W, device=depth.device)
        y = torch.linspace(.5 / H, 1 - .5 / H, H, device=depth.device)
        rays = depth.new_ones(B, V, 3, H, W)
        rays[:, :, 0] = (x - intr[..., :1, 2:]) / intr[..., :1, :1]
        rays[:, :, 1] = (y[:, None] - intr[..., 1:2, 2:]) / intr[..., 1:2, 1:2]
        c2w_r = svd_orthogonalize(self.c2w_r[view_ids])
        w2c = torch.cat((c2w_r.mT, c2w_r.mT @ -self.c2w_t[view_ids]), 2)
        w2c = torch.cat((w2c, self.w2c_0001.expand(w2c.shape[0], 1, 4)), 1)[None]
        return dict(
            mono_depth=data['input_depths'].view_as(depth),
            local_points=rays * depth,
            local_points_detach_xy=rays.detach() * depth,
            local_points_detach_z=rays * depth.detach(),
            depths=depth,
            camera_extrs=w2c
        )