from torch import nn, Tensor

from submodules.AnchorSplat.point_sampler import ClosePackedSampler  # ConsistentSampler
from submodules.AnchorSplat.point_net import PointBackbone
from submodules.AnchorSplat.gs_head import SplitGaussianHead
from submodules.AnchorSplat.renderer import Renderer


def affine_inverse(extrinsics: Tensor) -> Tensor:
    # return extrinsics.float().inverse().to(extrinsics.dtype)
    out = extrinsics.new_zeros(extrinsics.shape)
    out[..., :3, :3] = RT = extrinsics[..., :3, :3].mT
    out[..., :3, 3:] = RT @ -extrinsics[..., :3, 3:]
    out[..., 3, 3] = 1
    return out


def normalize_extrinsics(extrinsics: Tensor) -> tuple[Tensor, float]:
    '''Normalize input and view extrinsics according to translations'''
    extrinsics = extrinsics @ affine_inverse(extrinsics[..., :1, :, :])
    translations = affine_inverse(extrinsics)[..., :3, 3]
    median_dist = translations.norm(dim=-1).median().clamp(.1)
    extrinsics[..., :3, 3] /= median_dist
    return extrinsics


class AnchorSplatNet(nn.Module):
    '''
    AnchorSplat network for anchor-aligned GS prediction based on multi-view RGBD inputs.

    This network consists of:
    - Anchor sampler: voxel downsampling or TSDF fusion
    - 3D backbone: lightweight point Transformer
    - GS head: anchor-aligned GS prediction
    '''

    def __init__(self, anchor_sampler: dict, point_net: dict, gs_head: dict):
        super().__init__()
        self.anchor_sampler = ClosePackedSampler(**anchor_sampler)  # ConsistentSampler
        self.point_net = PointBackbone(**point_net)
        self.gs_head = SplitGaussianHead(**gs_head)
        self.renderer = Renderer()

    def forward(self, depth: Tensor, feature: Tensor, intrinsics: Tensor, extrinsics: Tensor,
                resolution: tuple[int, int]) -> Tensor:
        extrinsics = normalize_extrinsics(extrinsics)
        anchors, feature, voxel_sizes = self.anchor_sampler.forward(
            depth, feature, intrinsics, extrinsics)
        feature = self.point_net.forward(feature, anchors, voxel_sizes)
        num_kv = self.anchor_sampler.num_points[0]
        gs = self.gs_head.forward(feature[:, num_kv:], anchors[:, num_kv:], voxel_sizes)
        render = self.renderer.forward(gs, intrinsics, extrinsics, resolution)
        return render
