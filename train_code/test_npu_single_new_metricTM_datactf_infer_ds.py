# %%
import os
import torch
from torch.utils.data import DataLoader
torch._C._jit_set_profiling_mode(False)
from tqdm import tqdm

## opensora 
from opensora_local.utils.ckpt_utils import  load_checkpoint
from opensora_local.utils.config_utils import parse_configs

## src code
from src.registry import MODELS, build_module,build_dataset_kwargs
from src.dataset.dataset_build_scannet_v6 import DatasetForScanNetV6TESTSDF
from src.utils.misc import save_inout_pc_input,save_surface_as_mesh
from src.utils.metric import octree2volum,extract_surface_points
from src.models.utils import get_fine_voxel_mask_from_coarse_labels_fast

def expand_children_from_coarse_active(coarse_grid_active: torch.Tensor,
                                       resolution_coarse: int,
                                       scale_factor: int):
    """
    父层激活结点索引 -> 子层结点索引（不依赖 fine_grid）
    输入:
      coarse_grid_active: [Nc,3] int 索引 (范围 0..R_c)
      resolution_coarse:  R_c
      scale_factor:       s = vs_fine // vs_coarse (通常=2)
    输出:
      children: [Nf,3] int 子层索引 (范围 0..R_f, R_f = R_c * s)
      R_f:      子层分辨率
    """
    device = coarse_grid_active.device
    s = int(scale_factor)
    assert s >= 2 and (resolution_coarse >= 1)
    Rf = resolution_coarse * s

    rng = torch.arange(s, device=device, dtype=torch.int64)
    oz, oy, ox = torch.meshgrid(rng, rng, rng, indexing='ij')     # [s,s,s]
    offsets = torch.stack([oz, oy, ox], dim=-1).reshape(-1, 3)    # [s^3,3]

    parents = coarse_grid_active.to(torch.int64)                  # [Nc,3]
    children = parents[:, None, :] * s + offsets[None, :, :]      # [Nc,s^3,3]
    children = children.reshape(-1, 3)                            # [Nc*s^3,3]

    # 裁剪到 [0, Rf]
    valid = (children >= 0).all(dim=1) \
            & (children[:, 0] <= Rf) & (children[:, 1] <= Rf) & (children[:, 2] <= Rf)
    children = children[valid]
    # 如需去重: children = torch.unique(children, dim=0)
    return children, Rf

def gridindex_to_coords_node(grid_idx: torch.Tensor, vs: int):
    """
    将结点索引(0..vs)映射到连续坐标[-1,1]（与 voxel_size=2.0/vs 风格匹配）
    """
    step = 2.0 / float(vs)
    return -1.0 + grid_idx.to(torch.float32) * step

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=False)
    print("====== cfg: ",cfg)
    device = "npu:0"
    dtype = torch.float32
    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet    
    val_dataset = DatasetForScanNetV6TESTSDF(
        data_root=cfg.data_root,
        data_list_name=cfg.val_data_list_name,
        **build_dataset_kwargs(cfg.val_data_args)
    )

    dataloader = DataLoader(val_dataset,batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    # ======================================================
    # 3. exp output dir
    # ======================================================
    ckpt_path = os.path.join(cfg.root,cfg.ckpt_name)
    exp_name = os.path.join(cfg.outputs, cfg.ckpt_name, cfg.prefix)
    os.makedirs(exp_name, mode = 0o777, exist_ok = True)
    print("====== data saving to : ", exp_name)

    # ======================================================
    # 4. build model
    # ======================================================
    print("====== cfg.model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint(model, ckpt_path=ckpt_path)
    model.eval()

    # =======================================================
    # 5. Eval Loop (multi-resolution: e.g., [256, 512, 1024, ...])
    # =======================================================
    scale_levels = cfg.vs_res                  # e.g., [256, 512, 1024]
    assert isinstance(scale_levels, (list, tuple)) and len(scale_levels) >= 1
    num_levels = len(scale_levels)

    for input_dict in tqdm(dataloader):
        # move to device
        for k in list(input_dict.keys()):
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k][0].to(device)

        # 保存每层的输出 / 目标 / 掩码（第0层掩码为None）
        output_list, target_list, mask_list = [], [], [None]

        # 逐层细化需要的 latent
        latent = None

        # 为了保存 mesh
        sub_folder = input_dict['scene_id'][0]
        inout_dir = os.path.join(exp_name, sub_folder)
        os.makedirs(inout_dir, exist_ok=True)
        save_inout_pc_input(inout_dir, input_dict)

        print(f"\n================ Scene: {sub_folder} =================")
        # ---------- Level 0：最粗分辨率（用 full grid） ----------
        vs0 = scale_levels[0]
        with torch.no_grad():
            logit_0, target_0, latent = model.forward_ctf_infer(
                input_dict, vs0, latent=None, return_latent=True
            )
        output_list.append(logit_0)
        target_list.append(target_0)
        grid_cache = {vs0: input_dict[f'pc_query_grid_{vs0}']}

        n0 = grid_cache[vs0].shape[0]
        print(f"[Scene {sub_folder}] queries@{vs0}: {n0}")

        # 保存第0层网格（full grid）
        # print(f"\n***************[vs resolution: {vs0}]***************")
        pred_np_field_0 = octree2volum(
            logit_0.detach().cpu().numpy(),
            input_dict[f'pc_query_grid_{vs0}'].cpu().numpy(),
            tar_size=vs0, thres=cfg.truncatedist[0]
        )
        verts_0, faces_0 = extract_surface_points(
            pred_np_field_0, voxel_size=2.0/vs0, alpha=cfg.mc_alpha[0]
        )
        save_surface_as_mesh(f'{inout_dir}/pred_mesh_{vs0}.ply', verts_0, faces_0)

        # ---------- 后续各层：逐层细化（动态生成子节点，纯 inference） ----------
        for li in range(1, num_levels):
            vs_coarse = scale_levels[li-1]
            vs_fine   = scale_levels[li]

            # finer 尺度的阈值（保持你原策略）
            thr_label = 3.0 * 2.0 / vs_fine

            # 父层“实际用于推理的 grid”（第1层时是 full 256；之后是 partial 上一层）
            coarse_grid_used = grid_cache[vs_coarse]          # [Nc,3] 或 full
            logits_coarse_used = output_list[li-1]            # 与上面一一对应

            with torch.no_grad():
                label_coarse = (logits_coarse_used <= thr_label)

                # 取父层“激活”的结点索引
                active_coarse = coarse_grid_used[label_coarse.bool()]
                print(f"parent_active@{vs_coarse}: {active_coarse.shape[0]}")
                if active_coarse.numel() == 0:
                    print(f"[Warn] Empty active set at parent {vs_coarse} -> skip {vs_fine}.")
                    output_list.append(torch.empty(0, device=device))
                    target_list.append(torch.empty(0, device=device))
                    grid_cache[vs_fine] = coarse_grid_used[:0]
                    continue

                assert vs_fine % vs_coarse == 0, \
                    f"vs_fine ({vs_fine}) must be an integer multiple of vs_coarse ({vs_coarse})"
                scale_factor = vs_fine // vs_coarse

                # === 关键：父→子，直接生成 finer 层索引（无需 fine_grid_full）===
                child_idx, Rf = expand_children_from_coarse_active(
                    active_coarse, resolution_coarse=int(vs_coarse), scale_factor=int(scale_factor)
                )                                              # [Nf,3], Rf==vs_fine

                print(f"queries@{vs_fine}: {child_idx.shape[0]} (scale_factor={scale_factor})")

                # 生成 finer 的连续查询坐标（结点版）
                pc_query_fine = gridindex_to_coords_node(child_idx, vs=Rf)   # [Nf,3] float32

            # 构造 finer 的 partial 输入，仅在这些子结点做前向
            input_dict_fine = dict(input_dict)
            input_dict_fine[f'pc_query_{vs_fine}']      = pc_query_fine.to(device)
            input_dict_fine[f'pc_query_grid_{vs_fine}'] = child_idx.to(device)

            # 如果你的 model.forward_ctf_infer 需要 sdf 监督，这里是 test 脚本可不带:
            # if f'pc_query_sdf_{vs_fine}' in input_dict:
            #     ……（通常没有 partial 的 sdf，评估/保存 mesh 不需要）

            with torch.no_grad():
                logit_f_part, target_f_part = model.forward_ctf_infer(
                    input_dict_fine, vs_fine, latent=latent, return_latent=False
                )
            output_list.append(logit_f_part)
            target_list.append(target_f_part)

            # 作为下一层的“父”，缓存本层“实际推理过”的 partial grid
            grid_cache[vs_fine] = input_dict_fine[f'pc_query_grid_{vs_fine}']

            # === 保存该层的 mesh（partial grid + partial logits）===
            # print(f"\n***************[vs resolution: {vs_fine}]***************")
            pred_np_field_f = octree2volum(
                logit_f_part.detach().cpu().numpy(),
                input_dict_fine[f'pc_query_grid_{vs_fine}'].cpu().numpy(),
                tar_size=vs_fine, thres=cfg.truncatedist[li]
            )
            verts_f, faces_f = extract_surface_points(
                pred_np_field_f, voxel_size=2.0/vs_fine, alpha=cfg.mc_alpha[li]
            )
            save_surface_as_mesh(f'{inout_dir}/pred_mesh_{vs_fine}.ply', verts_f, faces_f)

                
if __name__ == "__main__":
    main()
    
    
    
