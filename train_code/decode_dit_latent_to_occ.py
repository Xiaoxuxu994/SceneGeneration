import os
import torch
import torch_npu
torch._C._jit_set_profiling_mode(False)
from tqdm import tqdm
import numpy as np
import math
## src code
from src.registry import MODELS, build_module
from src.models.vae import *
from opensora_local.utils.ckpt_utils import load_checkpoint_ckpt
from opensora_local.utils.config_utils import parse_configs

def genQuery(bbox, pix, dev, verbose=False):

    cx, cy, cz, lx, ly, lz = bbox
    
    # 计算边界
    x_min, x_max = cx - lx / 2, cx + lx / 2
    y_min, y_max = cy - ly / 2, cy + ly / 2
    z_min, z_max = cz - lz / 2, cz + lz / 2
    
    # 生成网格点1
    x_vals = torch.arange(x_min, x_max + pix, step=pix)
    y_vals = torch.arange(y_min, y_max + pix, step=pix)
    z_vals = torch.arange(z_min, z_max + pix, step=pix)
    
    # 生成笛卡尔积网格
    x_grid, y_grid, z_grid = torch.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # 合并网格点
    grid = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
    grid = grid.to(dev)

    gridlist = []
    if verbose:
        print(grid.shape[0], x_vals.shape[0], y_vals.shape[0], z_vals.shape[0], grid.shape[0] - x_vals.shape[0]*y_vals.shape[0]*z_vals.shape[0])
    N = (grid.shape[0] // 1400000) + 2
    step = grid.shape[0] // N
    start = 0
    for i in range(N):
        if i < N - 1 :
            gridlist.append(grid[start:(start+step),:])
        else:
            gridlist.append(grid[start:,:])
        start = start + step
    return gridlist, x_vals.shape[0], y_vals.shape[0], z_vals.shape[0] # 输出形状为 [N, 3]

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    print("====== cfg: ",cfg)
    device = "npu:0"
    dtype = torch.float32

    # ======================================================
    # 2. build model
    # ======================================================
    ckpt_path = os.path.join(cfg.root,cfg.ckpt_name)
    print("====== model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint_ckpt(model, ckpt_path, device=device)
    model.eval()

    # =======================================================
    # 3. Decode latent Sample
    # ======================================================
    print("=" * 30 + " start decode" + "=" * 30)
    latents = torch.load(cfg.latent_file) #[B,S,D]
    conds = torch.load(cfg.cond_file) #[B,1,d]
    if not cfg.save_split:
        occ_dict = {}
        if cfg.save_prob:
            occ_prob_dict = {}
    for i in tqdm(range(latents.shape[0]), desc="Decoding latents"):
        # 生成query
        latent = latents[i].unsqueeze(0).to(device)
        cond = conds[i].squeeze(0)
        query_list, nx,ny,nz = genQuery(cond, cfg.grid_size, device, verbose=cfg.verbose)
        if cfg.verbose:
            print(len(query_list))
        if cfg.save_prob:
            prob_list = []
        occupied_points_list = []
        # 生成占据场
        for j in range(len(query_list)):
            query = query_list[j]
            output = model.decode(latent, query.unsqueeze(0))
            output_logit = output.squeeze(-1).squeeze(0)
            prob = torch.sigmoid(output_logit).detach().cpu()
            occupied_points = query[prob >= cfg.occ_threshold]
            occupied_points = occupied_points.detach().cpu()
            if cfg.save_prob:
                prob_list.append(prob)
            occupied_points_list.append(occupied_points)
            # print(prob.shape, occupied_points.shape)   
        # 占据场的稀疏点写入文件
        occupied_points = torch.concat(occupied_points_list, dim=0)
        if cfg.save_split:
            occ_filename = cfg.occ_field_prefix + str(i) + '.pth' 
            torch.save(occupied_points, occ_filename)
        else:
            occ_dict[str(i)] = occupied_points
        # 占据场的完整网格值写入文件
        if cfg.save_prob:
            occ_prob = torch.concat(prob_list, dim=0).reshape(nx,ny,nz)
            occ_prob_filename = cfg.occ_prob_prefix + str(i) + '.pth'
            if cfg.save_split:
                torch.save(occ_prob, occ_prob_filename)
            else:
                occ_prob_dict[str(i)] = occ_prob
        if cfg.verbose:
            print(f'{i} latent decode finished.')
        torch_npu.npu.empty_cache()

    if not cfg.save_split:
        torch.save(occ_dict, cfg.occ_field_file)
        if cfg.save_prob:
            torch.save(occ_prob_dict, cfg.occ_prob_file)
if __name__ == "__main__":
    main()