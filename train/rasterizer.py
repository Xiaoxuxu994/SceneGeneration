import os
from typing import Dict, Optional, Tuple
from typing_extensions import Literal
import math
import torch
from torch import Tensor
from meta_gauss_render import calc_render, gs_sort, build_tile_gs_mask, projection_three_dims_gaussian_fused, get_render_schedule_cpp

# from meta_gauss_render import AscendGaussRender
# from meta_gauss_render.npu import CalcRender, get_render_schedule, get_num_vector_core

# from utils import eval_sh, validate_inputs
# from config import Config 
# from gsplat import (
#     _fully_fused_projection,
#     _quat_scale_to_covar_preci
# )

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit direction
    using hardcoded SH polynomials.
    Work with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1)**2]
        dirs: jnp.ndarrayunit directions [..., 3]
    Returns:
        [...,C]
    """
    assert deg <= 3 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[0] >= coeff

    result = C0 * sh[0, ...]
    if deg > 0:
        x, y, z = dirs[0:1, ...], dirs[1:2, ...], dirs[2:3, ...]
        result = (result - 
                C1 * y * sh[1, ...] + 
                C1 * z * sh[2, ...] - 
                C1 * x * sh[3, ...])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                + C2[0] * xy * sh[4, ...]
                + C2[1] * yz * sh[5, ...]
                + C2[2] * (2.0 * zz - xx - yy) * sh[6, ...]
                + C2[3] * xz * sh[7, ...]
                + C2[4] * (xx - yy) * sh[8, ...])

            if deg > 2:
                result = (result
                    + C3[0] * y * (3.0 * xx - yy) * sh[9, ...]
                    + C3[1] * xy * z * sh[10, ...]
                    + C3[2] * y * (4.0 * zz - xx - yy) * sh[11, ...]
                    + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12, ...]
                    + C3[4] * x * (4.0 * zz - xx - yy) * sh[13, ...]
                    + C3[5] * z * (xx - yy) * sh[14, ...]
                    + C3[6] * x * (xx - 3.0 * yy) * sh[15, ...])

    return result

def validate_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    render_mode: str,
    sh_degree: Optional[int],
    N: int,
    C: int
) -> None:
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    if sh_degree is None:
        # pass
        # treat colors as post-activation values, should be in shape [ N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

VERSION="meta_rasterize"
class Rasterizer:
    def __init__(self, tile_size=64, camera_model="pinhole") -> None:
        # self.cfg = cfg
        self.tile_size = tile_size
        self.camera_model=camera_model
    
    @torch.no_grad()
    def sort_gs(self, all_in_mask, depths):
        num_tile = self.tile_grid.shape[1]
        # print(f"num_tile : {num_tile}")
        # tile offset
        tile_offset = torch.sum(all_in_mask, dim=0).cumsum(dim=0)                         # [144] mask[N,144]先计算每个tile有重叠的球数，每个tile的球连续排布，所以每个tile的内存位置是球数累加，对齐？ 411 us
        # print(f"tile_offset : {tile_offset}")
        # print(f"torch_sums : {torch.sum(all_in_mask, dim=0)}")
        sorted_gs_ids = torch.zeros(tile_offset[-1], dtype=torch.int32, device=all_in_mask.device)     # 所有tiles的所有球数 615us
        for tile_id in range(num_tile):
                prev_offset = tile_offset[tile_id-1] if tile_id > 0 else 0                   # 当前tile的tile_offset，也就是当前tile前面有多少个球 56us
                tile_in_mask = all_in_mask[:, tile_id]
                tile_depths = depths[tile_in_mask]                                          # 获取对当前tile有影响的所有球的depth 589us 完全是index， 
                tile_gs_ids =tile_in_mask.nonzero()[:,0]                                         # 当前tile对应的所有球，根据其在所有球中的位置赋予id 712us nonzero400 to150
                _, local_sort_index = torch.sort(tile_depths, stable=True) 
                sorted_gs_ids[prev_offset:tile_offset[tile_id]]  = tile_gs_ids[local_sort_index] # 获取这些球在所有球当中的位置 129us index 
        
        return sorted_gs_ids, tile_offset

    def tile2image(self, rendered_image, height, width, channel_dim=3):
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size), math.ceil(self.padded_width/self.tile_size), self.tile_size, self.tile_size,-1)
        rendered_image = rendered_image.permute(0,2,1,3,4)
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size)*self.tile_size, math.ceil(self.padded_width/self.tile_size)*self.tile_size,-1)
        return rendered_image.permute(2,0,1)[:,:height,:width]
    
    def ascend_rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        splats: dict,
        colors,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # active_sh_degree = kwargs["sh_degree"] 
        # if getattr(self, 'ascend_render', None) is None:
            # self.ascend_render = AscendGaussRender(width=width, height=height, active_sh_degree=active_sh_degree,
            #                                         isect_mode='original', cpu_radix_sort=True)
            # self.ascend_render.tile_grid = self.ascend_render.tile_grid.to(splats["means"].device)
            # self.ascend_render.pix_coord = self.ascend_render.pix_coord.to(splats["means"].device)
        # if getattr(self, 'tile_size', None) is None:
        self.tile_size = self.tile_size
        self.padded_width = math.ceil(width/self.tile_size)*self.tile_size
        self.padded_height = math.ceil(height/self.tile_size)*self.tile_size
        self.padded_width = math.ceil(width/self.tile_size)*self.tile_size
        self.padded_height = math.ceil(height/self.tile_size)*self.tile_size
        self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, self.tile_size), \
                                                    torch.arange(0, self.padded_width, self.tile_size), indexing='ij'),dim=-1).view(-1,2).to(splats["means"].device).T.contiguous()
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width), torch.arange(self.padded_height), indexing='xy'), dim=-1).to(splats["means"].device)
        
        means = splats["means"]  # [N, 3]
        quats = splats["quats"]  # [N, 4]
        # scales = torch.exp(splats["scales"])  # [N, 3]
        # opacities = torch.sigmoid(splats["opacities"])  # [N,]
        scales = splats["scales"]
        opacities = splats["opacities"]

        # image_ids = kwargs.pop("image_ids", None) 
        # colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # sh0: [N, K, 3],  shN: [N, K, 3]

        rasterize_mode = "classic"
        render_colors, render_depth, info = self._ascend_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=camtoworlds,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            rasterize_mode=rasterize_mode,
            camera_model=self.camera_model,
            tile_size=self.tile_size,
            **kwargs,
        )
        return render_colors, render_depth, info

    def inverse_cov2d_v2(self, cov2_00, cov2_01, cov2_11, scale=1):
        det = cov2_00 * cov2_11 - cov2_01 * cov2_01
        inv_x_0 = cov2_11 / det * scale
        inv_x_1 = -cov2_01 / det * scale
        inv_x_2 = cov2_00 / det * scale
        return inv_x_0, inv_x_1, inv_x_2

    def _ascend_rasterization(
        self,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        eps2d: float = 0.3,
        sh_degree: Optional[int] = None,
        tile_size: int = 64,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor, Dict]:
        """A version of rasterization() that utilizes on PyTorch's autograd.
        Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

        .. note::
            This function relies on gsplat's CUDA backend for some computation, but the 
            entire differentiable graph is built with PyTorch (and nerfacc), so 
            back-propagation is handled by PyTorch's autograd.

        ..note::
            Compared to rasterization(), this function does not support some arguments such as 
            `packed`, `sparse_grad` and `absgrad`.
        """

        N = means.shape[0]
        C = viewmats.shape[0]
        B = 1
        validate_inputs(means, quats, scales, opacities, colors, viewmats, Ks, render_mode, sh_degree, N, C)

        ############## python 
        # Projection Gaussians to 2D
        # The results are with shape [C,N, ...]. Only the elements with radii > 0 are valid.
        # covars_py, _ = _quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False, triu=False)
        # radii_py, means2d_py, depths_py, conics_py, compensations_py, covars2d_py = _fully_fused_projection(
        #     means,
        #     covars_py,
        #     viewmats,
        #     Ks,
        #     width,
        #     height,
        #     eps2d=eps2d,
        #     near_plane=near_plane,
        #     far_plane=far_plane,
        #     calc_compensations=(rasterize_mode == "antialiased"),
        #     camera_model=camera_model
        # )

        if VERSION=="meta_rasterize":

            ############## ascendc 
            radii_new, means2d_new, depths_new, conics, compensations, filter = projection_three_dims_gaussian_fused(
                means.reshape(B,N,3).contiguous(),
                None,
                quats.reshape(B,N,4).contiguous(),
                scales.reshape(B,N,3).contiguous(),
                opacities.reshape(B,N).contiguous(),
                viewmats.reshape(B,C,4,4).contiguous(),
                Ks.reshape(B,C,3,3).contiguous(),
                width,
                height,
                eps2d,
                near_plane
            )
            radii = radii_new[0]
            means2d = means2d_new[0]
            depths = depths_new[0]

            radii = radii_new[filter].reshape(1,-1,2)
            means2d = means2d_new[filter].reshape(1,-1,2)
            depths = depths_new[filter].reshape(1,-1)
            opacities = opacities[filter[0,0]]
            conics = conics[filter].reshape(1,1,-1,3)
            colors = colors[filter[0,0]]

            # print(f"conics nan count : {torch.count_nonzero(conics.isnan())}")

            ############## python 
            # covars, _ = _quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False, triu=False)
            # radii, means2d, depths, conics, compensations, covars2d = _fully_fused_projection(
            #     means,
            #     covars,
            #     viewmats,
            #     Ks,
            #     width,
            #     height,
            #     eps2d=eps2d,
            #     near_plane=near_plane,
            #     far_plane=far_plane,
            #     calc_compensations=(rasterize_mode == "antialiased"),
            #     camera_model=camera_model
            # )
            # conics = conics.reshape(1,1,-1,3)
            # breakpoint()

        opacities = opacities # [C, N]
        camera_ids, gaussian_ids = None, None

        # if compensations is not None:
        #     opacities = opacities * compensations

        # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
        if sh_degree is None:
            # Colors are post-activation values, with shape [N, D] or [C, N, D]
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass
        else:
            # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
            camtoworlds = torch.inverse(viewmats) # [C, 4, 4]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)
            else:
                # colors is already [C, N, K, 3]
                shs = colors

            # spherical_harmonics的普通pytorch实现
            def build_color(means3D, shs, sh_degree, camera_center):
                rays_o = camera_center
                rays_d = means3D - rays_o
                rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
                color = eval_sh(sh_degree, shs.permute(1, 2, 0), rays_d.transpose(0, 1))
                color = (color+0.5).clip(min=0.0).transpose(0, 1)
                return color
            colors = build_color(means3D=means,
                                shs=shs[0],
                                sh_degree=sh_degree,
                                camera_center=camtoworlds[0, :3, 3])

        # ascend gauss render 排序
        with torch.no_grad():
            # all_in_mask = [
            #     # mask: [gs_num, tile_num]
            #     self.ascend_render.build_tile_gs_mask(
            #         means2d[_cam_view, :, 0], means2d[_cam_view, :, 1],
            #         radii[_cam_view].to(torch.int64).max(dim=1)[0], width, height,
            #         cov00=covars2d[_cam_view, :, 0, 0], cov01=covars2d[_cam_view, :, 0, 1], cov11=covars2d[_cam_view, :, 1, 1],
            #         opacity=opacities[_cam_view]) for _cam_view in range(C)
            # ]

            # print(f"341 radii: {radii}, torch.count_nonzero: {torch.count_nonzero(radii)}")
            if VERSION=="meta_rasterize":
                all_in_mask = [build_tile_gs_mask(
                    means2d[_cam_view, :, :2].T.contiguous(),
                    radii[_cam_view].to(torch.float32).max(dim=1)[0],
                    self.tile_grid,
                    width, height, self.tile_size)  for _cam_view in range(C)]

            sorted_gs_ids = []
            tile_offsets = []
            for _cam_view in range(C):
                # cf_sorted_gs_ids_py, cf_tile_offsets_py = self.sort_gs(
                #         all_in_mask[_cam_view].T.to(torch.bool), depths[_cam_view]
                #     )
                # print("cf_sorted_gs_ids_py", cf_sorted_gs_ids_py)
                if VERSION=="meta_rasterize":
                    cf_sorted_gs_ids_new, cf_tile_offsets_new, ret_code = gs_sort(all_in_mask[_cam_view].contiguous(), depths[_cam_view].contiguous())
                    # print("-0- cf_sorted_gs_ids_new", ret_code, cf_sorted_gs_ids_new)
                    if ret_code != 0:
                        cf_sorted_gs_ids_new, cf_tile_offsets_new = self.sort_gs(all_in_mask[_cam_view].T.to(torch.bool), depths[_cam_view])
                        # print("-1- cf_sorted_gs_ids_new", ret_code, cf_sorted_gs_ids_new)
                    # else:
                    #   cf_tile_offsets_new = cf_tile_offsets_new.to(torch.long)
                    # print("-2- cf_sorted_gs_ids_new", ret_code, cf_sorted_gs_ids_new, "cf_sorted_gs_ids_new.shape", cf_sorted_gs_ids_new.shape)
                sorted_gs_ids.append(cf_sorted_gs_ids_new)
                tile_offsets.append(cf_tile_offsets_new)

        render_colors = []
        render_depths = []
        for _cam_view in range(0, C):
            cf_means2 = means2d[_cam_view][:, :2].transpose(0, 1).contiguous()
            cf_opacity = opacities.contiguous()
            cf_colors3 = colors.transpose(0, 1).contiguous()
            # cf_conics0 = covars2d[_cam_view, :, 0, 0]
            # cf_conics1 = covars2d[_cam_view, :, 0, 1]
            # cf_conics2 = covars2d[_cam_view, :, 1, 1]
            # inv_x_0, inv_x_1, inv_x_2 = self.inverse_cov2d_v2(cf_conics0, cf_conics1, cf_conics2)
            inv_x_0 = conics[0,_cam_view,:,0]
            inv_x_1 = conics[0,_cam_view,:,1]
            inv_x_2 = conics[0,_cam_view,:,2]
            cf_depths = depths[_cam_view]

            padded_height = self.padded_height
            padded_width = self.padded_width
            pix_coords = self.pix_coord.reshape(padded_height//tile_size, tile_size, padded_width//tile_size, tile_size, 2) \
                .permute(0, 2, 1, 3, 4).reshape(padded_height//tile_size*padded_width//tile_size, tile_size*tile_size, 2) \
                .permute(0, 2, 1).to(torch.float32).contiguous()
            # nums: 每个tile对应的gs数量
            nums = torch.cat([tile_offsets[_cam_view][:1], tile_offsets[_cam_view][1:] - tile_offsets[_cam_view][:-1]]) 
            # lb_sched：[每个vector core要处理的tile数目的cumsum，依次对应的tile id，依次对应的tile offset]
            # lb_sched = torch.tensor(get_render_schedule(nums.cpu(), get_num_vector_core()), dtype=torch.int64, device=means2d.device) 
            lb_sched = torch.tensor(get_render_schedule_cpp(nums.cpu().to(torch.int64), 40), dtype=torch.int64, device=means2d.device) 

            # cf_render_colors, cf_render_depths = CalcRender.apply(cf_means2,
            #                                                 inv_x_0, inv_x_1, inv_x_2,
            #                                                 cf_opacity.squeeze(dim=-1),
            #                                                 cf_colors3,
            #                                                 cf_depths[None, :],
            #                                                 pix_coords,
            #                                                 lb_sched,
            #                                                 sorted_gs_ids[_cam_view])
            if VERSION=="meta_rasterize":
                # print(cf_means2.shape, cf_colors3.shape, cf_opacity.shape)
                cf_render_colors, cf_render_depths = calc_render(cf_means2.contiguous(),
                                                                 inv_x_0.contiguous(), inv_x_1.contiguous(), inv_x_2.contiguous(),
                                                                 cf_opacity.squeeze(dim=-1).contiguous(),
                                                                 cf_colors3.contiguous(),
                                                                 cf_depths[None, :].contiguous(),
                                                                 pix_coords.contiguous(),
                                                                 lb_sched.contiguous(),
                                                                 sorted_gs_ids[_cam_view].contiguous()
                                                                 )
            cf_render_colors = self.tile2image(cf_render_colors.permute(1, 2, 0), height, width)
            cf_render_depths = self.tile2image(cf_render_depths.permute(1, 2, 0), height, width)
            # 深度累加没有实际物理意义，需要除以权重alpha归一化得到人眼感知的期望深度
            # cf_last_cumsum = self.ascend_render.tile2image(cf_last_cumsum.unsqueeze(0).permute(1,2,0), tile_size=tile_size)
            # cf_render_alphas = 1 - torch.exp(cf_last_cumsum)
            # cf_render_depths = cf_render_depths / cf_render_alphas.clamp(min=1e-10) 

            render_colors.append(cf_render_colors.permute(1, 2, 0))
            render_depths.append(cf_render_depths.permute(1, 2, 0))
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)

        meta = {
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "tile_width": padded_width // tile_size,
            "tile_height": padded_height // tile_size,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
        return render_colors, render_depths, meta