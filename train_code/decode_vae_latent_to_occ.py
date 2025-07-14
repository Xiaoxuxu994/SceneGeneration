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
import pandas as pd
import json

def genQuery(bbox, pix, dev):

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
    # print(grid.shape[0], x_vals.shape[0], y_vals.shape[0], z_vals.shape[0], grid.shape[0] - x_vals.shape[0]*y_vals.shape[0]*z_vals.shape[0])
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
    file_list = os.listdir(cfg.latent_path)
    for _, file in enumerate(file_list):
        if file.rsplit('.')[1] != 'pth':
            file_list.remove(file)
    # load bounding box
    df = pd.read_csv(cfg.bbox_file)
    df.set_index("sha256", inplace=True)
    # decode all file
    for i, file in tqdm(enumerate(file_list), total=len(file_list), desc="Decoding latents"):
        sha256 = file.rsplit('.')[0]
        bbox = json.loads(df.loc[sha256, "bbox"])
        latent = torch.load(os.path.join(cfg.latent_path, file), map_location='cpu')
        # 生成query
        latent = latent.unsqueeze(0).to(device)
        query_list, nx,ny,nz = genQuery(bbox, cfg.grid_size, device)
        probs_list = []
        occupied_points_list = []
        # 生成占据场
        for j in range(len(query_list)):
            query = query_list[j]
            output = model.decode(latent, query.unsqueeze(0))
            output_logit = output.squeeze(-1).squeeze(0)
            probs = torch.sigmoid(output_logit).detach().cpu()
            occupied_points = query[probs >= cfg.occ_threshold]
            occupied_points = occupied_points.detach().cpu()
            probs_list.append(probs)
            occupied_points_list.append(occupied_points)
        probs = torch.concat(probs_list, dim=0)
        occupied_points = torch.concat(occupied_points_list, dim=0)
        occ_probs = probs.reshape(nx,ny,nz)
        # 占据场tensor写入文件
        torch.save(occupied_points, os.path.join(cfg.output_path, file))
        # torch.save(occ_probs, os.path.join(outputpath, 'occ_probs_' +  affix + '.pth'))
        tqdm.write(f'{i} latent decode finished.')
        torch_npu.npu.empty_cache()

if __name__ == "__main__":
    main()