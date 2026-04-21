from addict import Dict
import torch
from torch import Tensor
try:
    from submodules.AnchorSplat.utils.rasterizer import Rasterizer
    GS_VERSION = 'npu'
except ImportError:
    from gsplat import rasterization
    GS_VERSION = 'gsplat'


class Renderer(torch.nn.Module):
    '''
    3DGS render wrapper class that automatically selects from:
    - NPU: https://gitcode.com/cann/cann-recipes-spatial-intelligence/
    - GPU: https://github.com/nerfstudio-project/gsplat
    '''
    def __init__(self):
        super().__init__()
        if GS_VERSION == 'npu': self.rasterizer = Rasterizer()

    @torch.autocast(torch.accelerator.current_accelerator().type, enabled=False)
    def forward(self, gs: Dict[str, Tensor], intrinsics: Tensor, extrinsics: Tensor,
                resolution: tuple[int, int]) -> Tensor:
        for k in gs: gs[k] = gs[k].float()
        extrinsics, intrinsics = extrinsics.float(), intrinsics.float()
        if GS_VERSION == 'npu':
            render = []
            for i in range(intrinsics.shape[0]):
                # TODO: reuse projection forward and backward
                splats = {k: v[i] for k, v in gs.items()}
                for j in range(intrinsics.shape[1]):
                    image, depth = self.rasterizer.ascend_rasterize_splats(
                        w2c=extrinsics[i, j:j + 1],
                        Knorm=intrinsics[i, j:j + 1],
                        width=resolution[1],
                        height=resolution[0],
                        tile_size=32,
                        splats=splats,
                        active_sh_degree=0
                    )
                    render.append(torch.cat((image, depth), 1))
            render = torch.cat(render).unflatten(0, intrinsics.shape[:2])
        else:  # GS_VERSION == 'gsplat'
            scale = intrinsics.new_tensor((*resolution[::-1], 1.))[:, None]
            render = rasterization(
                gs.mean,
                gs.rotation,
                gs.scale,
                gs.opacity[..., 0],
                gs.color[..., None, :],
                extrinsics[None],
                intrinsics[None] * scale,
                resolution[1],
                resolution[0],
                near_plane=0.05,
                sh_degree=0,
                render_mode='RGB+ED'
            )[0].movedim(4, 2)
        return render
