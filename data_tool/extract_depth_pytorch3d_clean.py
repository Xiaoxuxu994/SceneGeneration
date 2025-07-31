import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
warnings.filterwarnings("ignore")
import torch
torch.backends.cuda.preferred_linalg_library("magma")
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 禁用 GUI，启用后台绘图
import matplotlib.pyplot as plt
import csv
import copy
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import gc
from torchvision.utils import save_image
import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRasterizer, 
)


DEBUG = False

import zipfile
import zlib
def check_integrity(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.testzip()  # 试图读取并验证压缩文件中的内容
        return True
    except zipfile.BadZipFile:
        print(f"[Warning] The file is corrupted or not a valid zip file. filepath is {file_path}")
        return False
    except zlib.error as e:
        # 捕获 zlib 解压缩错误，输出详细的错误信息
        print(f"[Warning] Zlib decompression error in file {file_path}. Error: {e}")
        return False
    except Exception as e:
        # 捕获其他可能的错误
        print(f"[Warning] An unexpected error occurred while checking file {file_path}. Error: {e}")
        return False
    
# %%
"""可视化utils"""        
def vis_uv(points):
    # 转为 numpy 以便 matplotlib 使用
    points_np = points.cpu().numpy()

    # 分离 x 和 y 坐标
    x = points_np[..., 0]
    y = points_np[..., 1]

    # 画图
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, c='blue', alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # 图像坐标系 y 轴向下，需要反转一下y轴    
    plt.grid(True)
    plt.axis("equal")  # 保持比例一致
    plt.savefig(f"{opt.output_dir}/uv_points_plot.png", dpi=300)  # 可以改为 .pdf / .svg / .jpg 等
    plt.close()  # 释放内存
                
def bilinear_sample_feat(feat_map, uv, debug=DEBUG):
    C, H, W = feat_map.shape
    x = uv[:, 0] / (W - 1) * 2 - 1
    y = uv[:, 1] / (H - 1) * 2 - 1
    
    grid = torch.stack([x, y], dim=1).unsqueeze(0).unsqueeze(0)  # [1,1,M,2]
    
    if debug:
        vis_uv(grid)
    
    feat_map = feat_map.unsqueeze(0)  # [1,C,H,W]
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # [1,C,1,M]
    sampled = sampled.squeeze(0).squeeze(2).T  # [M,C]
    return sampled

def vis_uv_in_depth(depth, uv, savepath='debug_projection.png'):
    if depth is not None:
        depth_map_np = depth.cpu().numpy()
        if depth_map_np.ndim == 3 and depth_map_np.shape[-1] == 1:
            depth_map_np = depth_map_np.squeeze(-1)
        uv_np = uv.cpu().numpy() if uv is not None else None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(depth_map_np, cmap='gray')
        if uv_np is not None and len(uv_np) > 0:
            ax.scatter(uv_np[:, 0], uv_np[:, 1], s=5, c='red', marker='o', alpha=0.6)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)  # ✅ 释放资源

def vis_depthmap(depthmap):
    plt.figure()
    plt.imshow(depthmap)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(opt.output_dir, f'batch{i}_depth.png'), bbox_inches='tight', pad_inches=-0.01)

# %%
def get_data(frames, sha256, new_w=1620, new_h=1080):

    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(opt.dataset_dir, 'renders', sha256, view['file_path'])
            try:
                image_ori = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image_ori = image_ori.resize((new_w, new_h), Image.Resampling.LANCZOS)  # NOTE: Scannet ： (1400, 840),
            image_ori = np.array(image_ori).astype(np.float32) / 255
            if image_ori.shape[2] > 3:
                image_ori = image_ori[:, :, :3] * image_ori[:, :, 3:]
            image_ori = torch.from_numpy(image_ori).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[2, :] *= -1
            c2w = c2w[np.array([1,0,2,3]), :]
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
                     
            is_bad = view['is_bad']
            
            # print(f"image: {image.shape, image.max()}, image_ori: {image_ori.shape, image_ori.max()}")
            
            return {
                'image_name': view['file_path'].split('.')[0],
                'ori_image_1080x1620': image_ori,
                'extrinsics': extrinsics,
                'is_bad': is_bad,
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data

def get_precise_T_norm(mesh, bbox_center, scale, R_ground=None, R_wall=None):
    if np.allclose(R_ground, np.eye(3)):
        center_ground = np.zeros(3)
    else:
        center_ground = mesh.get_center()
        
    if np.allclose(R_wall, np.eye(3)):
        center_wall = np.zeros(3)
    else:
        mesh.rotate(R_ground, center=center_ground)
        center_wall = mesh.get_center()
    
    # R1: rotate around center_ground
    T1 = np.eye(4)
    T1[:3, 3] = -center_ground
    R1_h = np.eye(4)
    R1_h[:3, :3] = R_ground
    T1_inv = np.eye(4)
    T1_inv[:3, 3] = center_ground
    Rg = T1_inv @ R1_h @ T1

    # R2: rotate around center_wall
    T2 = np.eye(4)
    T2[:3, 3] = -center_wall
    R2_h = np.eye(4)
    R2_h[:3, :3] = R_wall
    T2_inv = np.eye(4)
    T2_inv[:3, 3] = center_wall
    Rw = T2_inv @ R2_h @ T2

    # translation and scale
    T = np.eye(4)
    T[:3, 3] = -bbox_center
    S = np.eye(4)
    S[:3, :3] *= 1.0 / scale

    return S @ T @ Rw @ Rg

def compute_mesh_normalize(mesh,boxmin=-1,boxmax=1,boxscale=0.95):
    
    verts = np.array(mesh.vertices)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= center
    # scale = (boxmax - boxmin) / (verts.max(axis=0) - verts.min(axis=0)).max()
    scale = ((verts.max(axis=0) - verts.min(axis=0)).max()) / ((boxmax - boxmin) * boxscale)

    verts *= 1.0 / scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    
    # 归一化变换矩阵 M1
    M = np.eye(4)
    M[:3, :3] *= 1.0 / scale
    M[:3, 3] = -center / scale

    return mesh, M, scale


def get_norm_resize_intrinsics(fx, fy, cx, cy, W, H, resize_w, resize_h, scale):
    
    w_scale = resize_w / W
    h_scale = resize_h / H
    
    out_fx = fx * w_scale
    out_fy = fy * h_scale
    out_cx = cx * w_scale
    out_cy = cy * h_scale  
    
    if scale != 1: # norm
        
        out_fx = out_fx / scale  # fx
        out_fy = out_fy / scale  # fy
        # out_cx = (out_cx - cx) * scale_factor + cx  # 若有变化
        # out_cy = (out_cy - cy) * scale_factor + cy
    
    return out_fx, out_fy, out_cx, out_cy
    
def normalize_depth_global(depths, min_val=None, max_val=None):
    if min_val is None:
        min_val = depths.min()
    if max_val is None:
        max_val = depths.max()
    
    norm_depths = (depths - min_val) / (max_val - min_val + 1e-8)
    return norm_depths


import re
def extract_frame_id(path):
    """从路径中提取整数ID，例如 DSC0123.png -> 123"""
    match = re.search(r'DSC(\d+)', path)
    return int(match.group(1)) if match else -1

def find_nearby_frames(removed_ids, remaining_frames, window=20, step=4):
    """寻找与removed_ids在±window范围内的帧"""
    remaining_map = {extract_frame_id(f['file_path']): f for f in remaining_frames}
    selected = []
    selected_ids = set()

    # 构造对称偏移序列：4, -4, 8, -8, ..., window, -window
    offsets = []
    for i in range(1, (window // step) + 1):
        offsets.extend([i * step, -i * step])
    
    for rid in removed_ids:
        for offset in offsets:
            nid = rid + offset
            if nid in remaining_map and nid not in selected_ids:
                selected.append(remaining_map[nid])
                selected_ids.add(nid)
                break  # 只选一个最接近的
    return selected

# %%
def load_and_normalize_mesh(sha256, opt):
    o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(opt.dataset_dir, f'renders/{sha256}/mesh.ply'))
    # o3d_mesh_org = copy.copy(o3d_mesh)
    T_norm = None
    o3d_mesh, T_norm, scale = compute_mesh_normalize(o3d_mesh,boxmin=-1,boxmax=1,boxscale=0.95)
    SCALE = scale
    # o3d_mesh_org = o3d_mesh_org.transform(T_norm)
    # # # just do a quick check
    # o3d.io.write_triangle_mesh(os.path.join(opt.output_dir, f'{sha256}_meshnorm_direct.ply'), o3d_mesh)
    # o3d.io.write_triangle_mesh(os.path.join(opt.output_dir, f'{sha256}_meshnorm.ply'), o3d_mesh_org)

    vertices = torch.tensor(np.asarray(o3d_mesh.vertices), dtype=torch.float32).to("cuda")
    faces = torch.tensor(np.asarray(o3d_mesh.triangles), dtype=torch.int64).to("cuda")
    mesh = Meshes(verts=[vertices], faces=[faces])

    return mesh, o3d_mesh, torch.from_numpy(T_norm) if T_norm is not None else None, SCALE
        
def build_intrinsics(cam_infos, opt, h_hr=1080, w_hr=1620):
    fx, fy, cx, cy, W, H = cam_infos['fl_x'], cam_infos['fl_y'], cam_infos['cx'], cam_infos['cy'], cam_infos['w'], cam_infos['h']
    intrinsics = torch.tensor([[fx/W, 0.0, cx/W], [0.0, fy/H, cy/H], [0.0, 0.0, 1.0]])
    
    fx_hr, fy_hr, cx_hr, cy_hr = get_norm_resize_intrinsics(fx, fy, cx, cy, W, H, w_hr, h_hr, scale=1)  
    intrinsics_hr = torch.tensor([[fx_hr, 0.0, cx_hr], [0.0, fy_hr, cy_hr], [0.0, 0.0, 1.0]]) 
    
    rasterizer_hr = MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(
                image_size=(h_hr, w_hr),
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True,  # 更快，无影响
            )
        )
    
    return intrinsics, (fx, fy, cx, cy), (fx_hr, fy_hr, cx_hr, cy_hr), rasterizer_hr, intrinsics_hr

def render_depth(mesh, rasterizer_hr, batch_c2ws, T_norm, hr_cam_params, h_hr=1080, w_hr=1620):
    R_batch, T_batch = [], []              
    for idx, c2w in enumerate(batch_c2ws):
        # save_image(unnorm(batch_images[idx].unsqueeze(0))[0], 'images.png')  

        c2w = torch.inverse(c2w).float()  # 转换为相机坐标系
        T_norm_inverse = torch.inverse(T_norm).float().cuda()
        c2w = c2w @ T_norm_inverse
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        new_R = -R.T
        new_T = T.clone()
        new_T[:2] *= -1  # 翻转 x/y
        new_R[:, 2] *= -1  # 翻转 Z 轴

        R_batch.append(new_R)
        T_batch.append(new_T)
    R_batch = torch.stack(R_batch, dim=0)       # [B, 3, 3]
    T_batch = torch.stack(T_batch, dim=0)       # [B, 3]
    
    mesh_batch = mesh.extend(len(batch_c2ws))
    # 保存depth图
    fx_hr, fy_hr, cx_hr, cy_hr = hr_cam_params                                             
    # print(f"h_hr: {h_hr}, w_hr: {w_hr}, fx_hr: {fx_hr}, fy_hr: {fy_hr}, cx_hr: {cx_hr}, cy_hr: {cy_hr}")                                                
    cameras_hr = PerspectiveCameras(
                                    focal_length=[(fx_hr, fy_hr)] * len(batch_c2ws),
                                    principal_point=[(cx_hr, cy_hr)] * len(batch_c2ws),                       
                                    image_size=[(h_hr, w_hr)] * len(batch_c2ws),
                                    R=R_batch,
                                    T=T_batch,
                                    in_ndc=False,
                                    device=device
                                )
    # print("cameras_hr world position:", cameras_hr.get_camera_center())                        
    fragments_hr = rasterizer_hr(mesh_batch, cameras=cameras_hr)
    depthmap_hr = fragments_hr.zbuf[..., 0]/SCALE
    depthmap_hr = depthmap_hr.detach().cpu()    
    return depthmap_hr  


def render_depth_infer(mesh, rasterizer_hr, batch_c2ws, hr_cam_params, h_hr=1080, w_hr=1620):
    R_batch, T_batch = [], []              
    for idx, c2w in enumerate(batch_c2ws):
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        new_R = -R.T
        new_T = T.clone()
        new_T[:2] *= -1  # 翻转 x/y
        new_R[:, 2] *= -1  # 翻转 Z 轴

        R_batch.append(new_R)
        T_batch.append(new_T)
    R_batch = torch.stack(R_batch, dim=0)       # [B, 3, 3]
    T_batch = torch.stack(T_batch, dim=0)       # [B, 3]
    
    mesh_batch = mesh.extend(len(batch_c2ws))
    # 保存depth图
    fx_hr, fy_hr, cx_hr, cy_hr = hr_cam_params                                             
    # print(f"h_hr: {h_hr}, w_hr: {w_hr}, fx_hr: {fx_hr}, fy_hr: {fy_hr}, cx_hr: {cx_hr}, cy_hr: {cy_hr}")                                                
    cameras_hr = PerspectiveCameras(
                                    focal_length=[(fx_hr, fy_hr)] * len(batch_c2ws),
                                    principal_point=[(cx_hr, cy_hr)] * len(batch_c2ws),                       
                                    image_size=[(h_hr, w_hr)] * len(batch_c2ws),
                                    R=R_batch,
                                    T=T_batch,
                                    in_ndc=False,
                                    device=device
                                )
    # print("cameras_hr world position:", cameras_hr.get_camera_center())                        
    fragments_hr = rasterizer_hr(mesh_batch, cameras=cameras_hr)
    depthmap_hr = fragments_hr.zbuf[..., 0]/SCALE
    depthmap_hr = depthmap_hr.detach().cpu()    
    return depthmap_hr  

def save_pack_as_pt(pack: dict, save_path: str):
    pack_tensor = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            pack_tensor[k] = v
            print(f'{k}, {v.shape} \n')
        elif isinstance(v, np.ndarray):
            pack_tensor[k] = torch.from_numpy(v)
            print(f'{k}, {v.shape} \n')
        else:
            # 其他类型（字符串、数字、None等）直接保存
            pack_tensor[k] = v
            print(f'{k}, {v} \n')
    torch.save(pack_tensor, save_path)

if __name__ == '__main__':
    
    # python ./dataset_toolkits_scannet/extract_feature_correct_selectScenes_featUp_mp_fixed32.py --select_scenes c29b5e479c --batch_size 6 --case_id 3
    
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/data06/z84301856/projects/proj_3DGS/GPU_code/3DTokenizer/TRELLISdev/datasets/scannetpp_v2_opencv_pinhole/', # 'scannetpp_v2_pinhole', 'scannetpp_scene1'
                        help='Directory to save the metadata')
    parser.add_argument('--output_root', type=str, 
                        default='/data01/z00533370/dataset/scannetpp_volume_norm_-1_1_0.95_render_depth',
                        help='Directory to save the metadata')
    parser.add_argument('--select_scenes', nargs='+', type=str, default='0d2ee665be',
                        help='List of scenes to process')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=20)

    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    device = "cuda:0"
    opt.if_fixed32 = True
    opt.num_rend_views = 32
    INVALID_THRESHOLD = 0.2
    batch_size = opt.batch_size   
    vs_res = 128

    os.makedirs(opt.output_root, exist_ok=True)
    
    # get file list    
    with open("/data07/z00533370/dataset/scannetpp/splits/nvs_sem_val.txt") as f:
        sha256s = f.read().splitlines()

    # sha256s = ['0d2ee665be','1ada7a0617']
    with torch.no_grad():
        for sha256 in tqdm(sha256s, desc="Extracting features"):
            opt.output_dir = os.path.join(opt.output_root,sha256)
            os.makedirs(opt.output_dir, exist_ok=True)
            # 释放显存
            torch.cuda.empty_cache()
            gc.collect()
            
            cam_infos = json.load(open(os.path.join(opt.dataset_dir, 'renders', sha256, 'transforms.json'))) # copy folder
            all_frames = cam_infos['frames']
            all_frames = sorted(all_frames, key=lambda f: extract_frame_id(f['file_path']))
            
            mesh, o3d_mesh, T_norm, SCALE = load_and_normalize_mesh(sha256, opt)
            intrinsics, cam_params, hr_cam_params, rasterizer_hr, intrinsics_hr = build_intrinsics(cam_infos, opt)

            # generate franes
            # 确认fix还是shuffle
            if opt.if_fixed32:
                indices = np.linspace(0, len(all_frames) - 1, opt.num_rend_views, dtype=int)
                frames = [all_frames[idx] for idx in indices]
                print(f"[INFO] fixed 32: {len(frames)}")
                
            used_ids = set(f['file_path'] for f in frames)                
            remaining_frames = [f for f in all_frames if f['file_path'] not in used_ids]
            
            data = []
            for datum in get_data(frames, sha256):
                datum['intrinsics'] = intrinsics
                datum['resize_intrinsics'] = intrinsics_hr
                data.append(datum)

            all_depths, all_ori_images = [], []
            all_Tnorm_extrinsic, all_extrinsic, all_normed_intrinsic = [], [], []
            all_filenames = []
            all_resize_intrinsics = []
            
            max_valid_views = opt.num_rend_views  # 例如 32
            max_trials = 20     # 最多尝试 10 倍
            num_trials = 0
            total_valid_frames = 0
            current_index = 0
            while current_index < len(data) and total_valid_frames < max_valid_views and num_trials < max_trials:
                batch_data = data[current_index : current_index+batch_size]
                real_batch_size = min(batch_size, len(batch_data))
                if len(batch_data) == 0:
                    print("[WARN] No more batch data.")
                    break         
                
                bs = len(batch_data)
                batch_filenames = [d['image_name'] for d in batch_data]
                batch_ori_images = torch.stack([d['ori_image_1080x1620'] for d in batch_data])             
                batch_resize_intrinsics = torch.stack([d['resize_intrinsics'] for d in batch_data])  
                T_norm_inverse = torch.inverse(T_norm).float().cuda()
                batch_ori_extrinsics = torch.stack([d['extrinsics'] for d in batch_data])     
                batch_extrinsics = torch.stack([d['extrinsics'].to(device)@T_norm_inverse for d in batch_data])
                batch_extrinsics = batch_extrinsics.cuda()                                        
                batch_c2ws = torch.stack([torch.inverse(d['extrinsics']) for d in batch_data]).cuda()
                  
                batch_ori_intrinsics = torch.stack([d['intrinsics'] for d in batch_data])
                
                depthmap_hr = render_depth(mesh, rasterizer_hr, batch_c2ws, T_norm, hr_cam_params)
                print(f"[INFO] finished render depthmaps. ",
                        f"Depthmap_hr: {depthmap_hr.shape, depthmap_hr.min(), depthmap_hr.max()}; ")
                
                # ====== MODIFIED: 筛除无效帧 ======
                valid_indices = []
                for j in range(depthmap_hr.shape[0]):
                    invalid_mask = (depthmap_hr[j] <= 1e-4)  # 设为无效深度阈值
                    invalid_ratio = invalid_mask.sum().item() / invalid_mask.numel()
                    fid = extract_frame_id(batch_filenames[j])
                    if invalid_ratio < INVALID_THRESHOLD:
                        valid_indices.append(j)
                        status = 'kept'
                    else:
                        status = 'dropped'
                        print(f"[WARN] dropped frame {batch_filenames[j]} due to {invalid_ratio:.2f} invalid depth")

                # 保留有效帧                
                batch_data = [batch_data[i] for i in valid_indices]
                
                # ====== 找出被剔除帧的 ID ======
                removed_indices = [j for j in range(bs) if j not in valid_indices]
                removed_ids = [extract_frame_id(batch_filenames[j]) for j in removed_indices]
                print(f"[INFO] Batch {current_index // batch_size}: Removed {len(removed_ids)} frames")
                # ====== 日志原始shape ======
                print(f"  - Original batch size: {bs}")
                print(f"  - Valid frames: {len(valid_indices)}")
                print(f"  - depthmap_hr: {depthmap_hr.shape}")
                print(f"  - batch_extrinsics: {batch_extrinsics.shape}")
                
                # ====== 找到附近帧并补充 ======
                if remaining_frames and len(removed_ids) > 0:
                    window = 100 if len(remaining_frames) > 2000 else 50
                    candidates = find_nearby_frames(removed_ids, remaining_frames, window=window, step=4)
                    # print(f"  - Attempting to replace {len(removed_ids)} dropped frames with {len(candidates)} nearby candidates...")
                    # print(f"&&& len(candidates): {len(candidates)}, len(batch_data): {len(batch_data)}, used_ids: {used_ids}")
                    for new_frame in candidates:
                        if len(batch_data) >= real_batch_size:
                            break
                        if new_frame['file_path'] in used_ids:
                            continue
                        used_ids.add(new_frame['file_path'])
                        remaining_frames.remove(new_frame)
                        for new_data in get_data([new_frame], sha256):
                            new_data['intrinsics'] = intrinsics
                        data.append(new_data)
                        print(f"    -> Replaced with: {new_frame['file_path']} (ID: {extract_frame_id(new_frame['file_path'])})")
                        new_fname = os.path.basename(new_frame['file_path'])
                        new_fid = extract_frame_id(new_fname)
                
                # 更新 batch_size 日志
                print(f"  - Final batch size: {len(batch_data)}, replacement frames: {len(removed_ids)}\n")
                
                if len(valid_indices) > 0:
                    batch_ori_images = batch_ori_images[valid_indices]
                    depthmap_hr = depthmap_hr[valid_indices]
                    batch_extrinsics = batch_extrinsics[valid_indices]
                    batch_c2ws = batch_c2ws[valid_indices]
                    batch_ori_intrinsics = batch_ori_intrinsics[valid_indices]
                    batch_filenames = [batch_filenames[i] for i in valid_indices]  
                    batch_resize_intrinsics = batch_resize_intrinsics[valid_indices]

                    all_Tnorm_extrinsic.append(batch_extrinsics)
                    all_resize_intrinsics.append(batch_resize_intrinsics)
                    all_extrinsic.append(batch_ori_extrinsics[valid_indices])

                    all_ori_images.append(batch_ori_images)
                    all_normed_intrinsic.append(batch_ori_intrinsics)
                    all_depths.append(depthmap_hr)
                    all_filenames.extend(batch_filenames)
                    
                total_valid_frames += len(valid_indices)
                print(f"[INFO] total_valid_frames = {total_valid_frames}\n")
                current_index += real_batch_size
                num_trials += 1
                if total_valid_frames >= max_valid_views:
                    print(f"[INFO] Reached max valid frames: {total_valid_frames}, stopping early.")
                    break
                if num_trials >= max_trials:
                    print(f"[WARN] Exceeded max trials ({max_trials}), stopped with only {total_valid_frames} valid frames.")
                    break
            
            all_resize_intrinsics = torch.cat(all_resize_intrinsics, dim=0)[:max_valid_views]
            all_depths = torch.cat(all_depths, dim=0)[:max_valid_views]
            all_ori_images = torch.cat(all_ori_images, dim=0)[:max_valid_views]
            all_extrinsic = torch.cat(all_extrinsic, dim=0)[:max_valid_views]
            all_Tnorm_extrinsic = torch.cat(all_Tnorm_extrinsic, dim=0)[:max_valid_views]
            all_normed_intrinsic = torch.cat(all_normed_intrinsic, dim=0)[:max_valid_views]
            all_filenames = all_filenames[:max_valid_views]
            
            pack = {}
            pack['scene_id'] = sha256  # 字符串也可以存

            # pack['T_norm'] = T_norm
            pack['scale'] = SCALE
            pack[f'extrinsics_Tnorm_norm_{vs_res}'] = all_Tnorm_extrinsic                
            pack[f'depthmap_1080x1620_norm_{vs_res}'] = all_depths
            # pack[f'normed_intrinsics_norm_{vs_res}'] = all_normed_intrinsic
            # pack[f'ori_extrinsics_norm_{vs_res}'] = all_extrinsic
            # pack[f'images_1080x1620_{vs_res}'] = all_ori_images
            pack[f'resized_intrinsics_norm_{vs_res}'] = all_resize_intrinsics[0,:,:]
            
            save_path = os.path.join(opt.output_dir, f'{sha256}.pt')
            save_pack_as_pt(pack, save_path)
            print(f'save all results (positions, depths, images, caminfos, agg_feats, filled_agg_feats) to : {save_path}') 
            save_image(all_ori_images, os.path.join(opt.output_dir, f'{sha256}_usedImgs.png'))  
            save_image(all_depths.unsqueeze(1), os.path.join(opt.output_dir, f'{sha256}_usedDepths.png'))  
            gc.collect()    
            torch.cuda.empty_cache()     
