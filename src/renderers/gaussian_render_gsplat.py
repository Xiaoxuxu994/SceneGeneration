#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Union
import torch
import math
import time
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict
from gsplat.rendering import rasterization
import time


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def render(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    rendering_mode = 'gsplat'
    
    # lazy import
    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if rendering_mode == 'diff_gaussian':
        kernel_size = pipe.kernel_size
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return edict({"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii})


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        # camera = torch.inverse(view)[:3, 3]
        view_device = view.device
        view_cpu = view.to("cpu")
        camera_cpu = torch.inverse(view_cpu)[:3, 3]
        camera = camera_cpu.to(view_device)

        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        if isinstance(resolution, int):
            resolution_w, resolution_h = resolution, resolution
        elif isinstance(resolution, tuple) or isinstance(resolution, list):
            resolution_h, resolution_w = resolution[0], resolution[1]
        else:
            raise RuntimeError("resolution is not int, tuple, list!")
        camera_dict = edict({
            "image_height": resolution_h * ssaa,
            "image_width": resolution_w * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

        # Render
        render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution_h, resolution_w), mode='bilinear', align_corners=False, antialias=True).squeeze()
            
        ret = edict({
            'color': render_ret['render']
        })
        return ret

# def render_gsplat(gausssian: Gaussian, 
#                   extrinsics: torch.Tensor,
#                   intrinsics: torch.Tensor,
#                   resolution: tuple,
#                   render_mode: str,
#                   sh_degree: int = 0,
#                   AbsGrad: bool = False):

#     means3D = gausssian.get_xyz #.float()
#     opacity = gausssian.get_opacity # .float()
#     scales = gausssian.get_scaling # .float()
#     rotations = gausssian.get_rotation #.float()
#     shs = gausssian.get_features #.float()
#     shs = shs.view(means3D.shape[0], -1, 3)

#     if len(extrinsics.shape) == 4 and len(intrinsics.shape) == 4:
#         extrinsics = extrinsics.squeeze(0)
#         intrinsics = intrinsics.squeeze(0)

#     # torch.cuda.reset_peak_memory_stats()
#     # torch.cuda.synchronize()
#     # start_time = time.time()
    
#     renders_images, _, _ = rasterization(
#             means = means3D,  # [N, 3]
#             quats = rotations,  # [N, 4]
#             scales = scales,  # [N, 3]
#             opacities = opacity.squeeze(),  # [N]
#             sh_degree=sh_degree,
#             colors = shs,  # [N, S, 3]
#             viewmats = extrinsics,  # [C, 4, 4] # 。squeeze(0)
#             Ks = intrinsics,  # [C, 3, 3] # .squeeze(0)
#             width = resolution[1],
#             height = resolution[0],
#             render_mode=render_mode,
#             packed=False,
#             absgrad=AbsGrad
#             # distributed=world_size > 1,
#         )

#     # torch.cuda.synchronize()
#     # end_time = time.time()
#     # peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # 转为 MB
#     # print(f"[Profiling] img resolution: {resolution}, Time: {end_time - start_time:.3f}s, Peak GPU Mem: {peak_mem:.2f} MB")
        
#     # data = {
#     #             'xyz': means3D.detach().cpu().numpy(),
#     #             'opacity': opacity.squeeze().detach().cpu().numpy(),
#     #             'scales': scales.detach().cpu().numpy(),
#     #             'rotations': rotations.detach().cpu().numpy(),
#     #             'shs': shs.detach().cpu().numpy(),
#     #             'extrinsics': extrinsics.detach().cpu().numpy(),
#     #             'intrinsics': intrinsics.detach().cpu().numpy(),
#     #             'rend_images': renders_images[..., 0:3].detach().cpu().numpy(),
#     #         }
#     # np.savez('gaussian_params.npz', **data)
    

#     if renders_images.shape[-1] == 4:
#         colors, depths = renders_images[..., 0:3], renders_images[..., 3:4]
#     else:
#         colors, depths = renders_images, None

#     if depths is None:
#         ret = edict({
#             'color': colors
#             })
#     else:
#         ret = edict({
#                 'color': colors,
#                 'depth': depths,
#             })
#     return ret


def render_gsplat(gausssian: Union[Gaussian, torch.Tensor], 
                  extrinsics: torch.Tensor,
                  intrinsics: torch.Tensor,
                  resolution: tuple,
                  render_mode: str,
                  sh_degree: int = 0,
                  AbsGrad: bool = False):
    if isinstance(gausssian, torch.Tensor):
        means3D = gausssian[:,:3]
        opacity = gausssian[:, 3:4]
        scales = gausssian[:, 4:7]
        rotations = gausssian[:, 7:11]
        shs = gausssian[:, 11:]
    else:
        means3D = gausssian.get_xyz #.float()
        opacity = gausssian.get_opacity # .float()
        scales = gausssian.get_scaling # .float()
        rotations = gausssian.get_rotation #.float()
        shs = gausssian.get_features #.float()
    shs = shs.view(means3D.shape[0], -1, 3)

    if len(extrinsics.shape) == 4 and len(intrinsics.shape) == 4:
        extrinsics = extrinsics.squeeze(0)
        intrinsics = intrinsics.squeeze(0)

    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()
    # start_time = time.time()
    
    renders_images, _, _ = rasterization(
            means = means3D,  # [N, 3]
            quats = rotations,  # [N, 4]
            scales = scales,  # [N, 3]
            opacities = opacity.squeeze(),  # [N]
            sh_degree=sh_degree,
            colors = shs,  # [N, S, 3]
            viewmats = extrinsics,  # [C, 4, 4] # 。squeeze(0)
            Ks = intrinsics,  # [C, 3, 3] # .squeeze(0)
            width = resolution[1],
            height = resolution[0],
            render_mode=render_mode,
            packed=False,
            absgrad=AbsGrad,
            near_plane = 1e-10, # NOTE: NEW
            eps2d = 0.01, # NOTE: NEW
            rasterize_mode='classic', # NOTE: NEW
            # distributed=world_size > 1,
        )

    # torch.cuda.synchronize()
    # end_time = time.time()
    # peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # 转为 MB
    # print(f"[Profiling] img resolution: {resolution}, Time: {end_time - start_time:.3f}s, Peak GPU Mem: {peak_mem:.2f} MB")
        
    # data = {
    #             'xyz': means3D.detach().cpu().numpy(),
    #             'opacity': opacity.squeeze().detach().cpu().numpy(),
    #             'scales': scales.detach().cpu().numpy(),
    #             'rotations': rotations.detach().cpu().numpy(),
    #             'shs': shs.detach().cpu().numpy(),
    #             'extrinsics': extrinsics.detach().cpu().numpy(),
    #             'intrinsics': intrinsics.detach().cpu().numpy(),
    #             'rend_images': renders_images[..., 0:3].detach().cpu().numpy(),
    #         }
    # np.savez('gaussian_params.npz', **data)
    

    if renders_images.shape[-1] == 4:
        colors, depths = renders_images[..., 0:3], renders_images[..., 3:4]
    else:
        colors, depths = renders_images, None

    if depths is None:
        ret = edict({
            'color': colors
            })
    else:
        ret = edict({
                'color': colors,
                'depth': depths,
            })
    return ret