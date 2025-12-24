import torch
import numpy as np
from tqdm import tqdm
# import utils3d
from PIL import Image
import time

# from ..renderers import OctreeRenderer, MeshRenderer
from ..renderers import render_metagauss_perview #, SurfelGaussianRenderer
from ..representations import Gaussian #, MeshExtractResult
from ..modules import sparse as sp


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(gauss_render, sample, extrinsics, intrinsics, options={}, 
                  is_infer=False, 
                  render_mode='RGB+D', 
                  render_option='meta_gauss_render',
                  **kwargs):
    
    if render_option == 'meta_gauss_render':
        
        rets = {'color': None, 'depth': None}
        if is_infer == True:
            # 推理渲染detach
            
            res = render_metagauss_perview(gauss_render, sample, extrinsics, intrinsics, 
                                            resolution=options.get('resolution', (512, 512)),
                                           output_depth=('D' in render_mode) )
            rets['color'] = res['render_image'].detach().clamp_(0, 1)
            if 'render_depth' in res:
                rets['depth'] = res['render_depth'].detach()
        else:
            # 渲染部分打开训练
            res = render_metagauss_perview(gauss_render, sample, extrinsics, intrinsics, 
                                           resolution=options.get('resolution', (512, 512)), 
                                           output_depth=('D' in render_mode) )
            rets['color'] = res['render_image'].clamp_(min=0.0, max=1.0)
            if 'render_depth' in res:
                rets['depth'] = res['render_depth'] # [1,H,W]
        return rets
    
    else:
        raise NotImplementedError
    
# def render_video(gauss_render, sample, 
#                  resolution=(512,512), 
#                  bg_color=(0, 0, 0), 
#                  num_frames=300, 
#                  r=2, fov=40, **kwargs):
#     yaws = torch.linspace(0, 2 * 3.1415, num_frames)
#     # pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))    
#     pitch = 0.15 * torch.cos(torch.linspace(0, 2 * 3.1415, num_frames))    
#     yaws = yaws.tolist()
#     pitch = pitch.tolist()
#     extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
#     flag_infer = True
#     return render_frames(gauss_render, sample, extrinsics, intrinsics, 
#                          options={'resolution': resolution, 'bg_color': bg_color}, 
#                          is_infer=flag_infer, **kwargs)

# def render_snapshot(gauss_render, samples, 
#                     resolution=(512,512), 
#                     bg_color=(0, 0, 0), 
#                     offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), 
#                     r=10, fov=8, **kwargs):
#     yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
#     yaw_offset = offset[0]
#     yaw = [y + yaw_offset for y in yaw]
#     pitch = [offset[1] for _ in range(4)]
#     extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
#     return render_frames(gauss_render, samples, extrinsics, intrinsics, 
#                          options={'resolution': resolution, 'bg_color': bg_color}, 
#                          **kwargs)

# def render_pose_infer(gauss_render, sample, 
#                       resolution=(512,512), 
#                       bg_color=(0, 0, 0), 
#                       extrinsics=None, 
#                       intrinsics=None,  
#                       render_mode="RGB+D", 
#                       **kwargs):
#     flag_infer = True
#     return render_frames(gauss_render, sample, extrinsics, intrinsics, 
#                          options={'resolution': resolution, 'bg_color': bg_color}, 
#                          is_infer=flag_infer, 
#                          render_mode=render_mode, 
#                          **kwargs)

def render_gtpose(gauss_render, sample, 
                  resolution=(512,512),
                  bg_color=(0, 0, 0),  
                  extrinsics=None, 
                  intrinsics=None, 
                  is_infer=False,
                  render_mode="RGB+D", 
                  **kwargs):
    return render_frames(gauss_render, sample, extrinsics, intrinsics, 
                         options={'resolution': resolution, 'bg_color': bg_color},  
                         is_infer=is_infer, 
                         render_mode=render_mode, 
                         **kwargs)