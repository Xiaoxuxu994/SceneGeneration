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
from src.dataset.dataset_build_scannet_v6 import DatasetForScanNetV6DEBUGAUGSDF
from src.utils.misc import save_inout_pc_input,save_surface_as_mesh
from src.utils.metric import octree2volum,extract_surface_points,sdf_eval_iou,init_running_metrics,udf_l1_l2_metric,sdf_eval_metric_all_TM
import numpy as np 
import re 
from src.models.utils import get_fine_voxel_mask_from_coarse_labels_fast

def log_metrics(metrics, step_count):
    avg_metrics = {k: v / step_count for k, v in metrics.items()}

    default_keys = [
        'IoU_cl1', 'IoU_cl0', 'IoU_wavg',
        'Precision_cl1', 'Precision_cl0', 'Precision_wavg',
        'Recall_cl1', 'Recall_cl0', 'Recall_wavg',
        'F1_cl1', 'F1_cl0', 'F1_wavg',
        'SDF_L1','SDF_L2']
    
    level_groups = {}
    default_metrics = {}

    for k, v in avg_metrics.items():
        # 仅保留前缀匹配的项
        if not any(k.startswith(prefix) for prefix in default_keys):
            continue

        # 分类为多层级还是默认
        match = re.fullmatch(r'(.*)_s(\d+)', k)
        if match:
            name, scale = match.group(1), match.group(2)
            scale = f's{scale}'
            if scale not in level_groups:
                level_groups[scale] = {}
            level_groups[scale][name] = v
        else:
            default_metrics[k] = v

    return level_groups, default_metrics

# def fill_missing_as_negative(pred_logits_partial, mask, total_voxel_num):
#     """
#     Args:
#         pred_logits_partial: (M,) float tensor, logits on selected query points
#         mask: (N,) bool tensor, which voxels in full set were selected (M==sum(mask))
#         total_voxel_num: int, N

#     Returns:
#         full_logits: (N,) float tensor, where unselected ones are set to -inf
#     """
#     full_logits = torch.full((total_voxel_num,), float('-inf'), device=pred_logits_partial.device)
#     full_logits[mask] = pred_logits_partial
#     return full_logits

def fill_missing_as_negative(pred_logits_partial, mask, total_voxel_num):
    """
    Args:
        pred_logits_partial: (M,) float tensor
        mask: (N,) bool tensor  or (M,) long indices 
        total_voxel_num: int, N
    Returns:
        full_logits: (N,) float tensor, 未选中的位置填 -inf
    """
    # -------- 简单缓存：按 (device, dtype, N) 复用大缓冲，避免每次分配 --------
    # 函数属性缓存（首次调用时创建）
    if not hasattr(fill_missing_as_negative, "_buf_cache"):
        fill_missing_as_negative._buf_cache = {}  # {(device, dtype, N): tensor}

    dev  = pred_logits_partial.device
    dtype= pred_logits_partial.dtype
    key  = (dev, dtype, int(total_voxel_num))

    full_logits = fill_missing_as_negative._buf_cache.get(key)
    if (full_logits is None) or (full_logits.numel() != total_voxel_num) \
       or (full_logits.device != dev) or (full_logits.dtype != dtype):
        full_logits = torch.empty((total_voxel_num,), device=dev, dtype=dtype)
        fill_missing_as_negative._buf_cache[key] = full_logits

    # 先整体置为 -inf（原地，不新分配）
    full_logits.fill_(float('inf'))

    # -------- 用 long 索引写入，比 bool 切片更快 --------
    if mask.dtype == torch.bool:
        # 长度检查（便于早发现问题）
        if mask.numel() != total_voxel_num:
            raise IndexError(f"mask numel {mask.numel()} != total_voxel_num {total_voxel_num}")
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    elif mask.dtype in (torch.long, torch.int64, torch.int32):
        idx = mask.to(torch.long)
    else:
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")

    # 形状一致性（M）
    if idx.numel() != pred_logits_partial.numel():
        raise IndexError(f"index length {idx.numel()} != logits length {pred_logits_partial.numel()}")

    # 原地写入（不累加）
    full_logits.index_put_((idx,), pred_logits_partial, accumulate=False)
    return full_logits

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
    val_dataset = DatasetForScanNetV6DEBUGAUGSDF(
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
    # 5. Eval Loop
    # =======================================================
    scale_level = cfg.vs_res
    vs_c = scale_level[0]
    vs_f = scale_level[-1]

    running_eval_metrics = init_running_metrics(levels=scale_level,
                            class_keys=['cl1','cl0','wavg'],
                            base_metrics=['Precision','Recall','F1','IoU'],
                            include_per_level=True)
       
    for input_dict in tqdm(dataloader):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key][0].to(device)

        output_list, target_list, mask_list = [], [], [None]

        fine_grid_full = input_dict['pc_query_grid_512']      
        fine_total = fine_grid_full.shape[0] 

        with torch.no_grad():
            # ========== 第1轮 vs256 ==========
            logit_c, target_c, latent = model.forward_ctf_infer(
                input_dict, vs_c, latent=None, return_latent=True
            )
            output_list.append(logit_c)
            target_list.append(target_c)

            # 生成 coarse->fine 的布尔掩码（长度=512全网格长度） # <<<
            label_c = (logit_c <= 3.0 * 2.0 / 512)
            grid_c = input_dict['pc_query_grid_256']
            fine_grid = fine_grid_full                         
            mask = get_fine_voxel_mask_from_coarse_labels_fast(
                coarse_label=label_c,
                scale_factor=2,
                resolution_coarse=256,
                coarse_grid=grid_c,
                fine_grid=fine_grid
            )
            mask_list.append(mask)  

        input_dict_fine = dict(input_dict)
        input_dict_fine['pc_query_512'] = input_dict['pc_query_512'][mask]
        input_dict_fine['pc_query_sdf_512'] = input_dict['pc_query_sdf_512'][mask]
        input_dict_fine['pc_query_grid_512'] = fine_grid_full[mask]

        with torch.no_grad():
            # ========== 第2轮 vs512（部分）==========
            logit_f_part, target_f_part = model.forward_ctf_infer(
                input_dict_fine, vs_f, latent=latent, return_latent=False
            )
            output_list.append(logit_f_part)
            target_list.append(target_f_part)

        # === save and print ===
        sub_folder = input_dict['scene_id'][0]
        inout_dir = os.path.join(exp_name, sub_folder)
        os.makedirs(inout_dir, exist_ok=True)
        save_inout_pc_input(inout_dir, input_dict)

        result_txt_path = os.path.join(inout_dir, 'result.txt')
        print(f"\n================ Scene: {sub_folder} =================")
        with open(result_txt_path, 'w') as f_s:
            print('================ Per-scene Evaluation =================', file=f_s)
            print(f"Scene: {sub_folder}")
            print(f"Scene: {sub_folder}", file=f_s)

            for idx, (logit_i, mask_i, target_i) in enumerate(zip(output_list, mask_list, target_list)):
                scale = scale_level[idx]

                print(f"\n***************[vs resolution: {scale}]***************")
                print(f"\n***************[vs resolution: {scale}]***************", file=f_s)

                if scale == vs_c:
                    # 第1轮：全量
                    metrics = sdf_eval_metric_all_TM(
                        logit_i, input_dict[f'pc_query_sdf_{scale}'],
                        sdf_threshold=cfg.sdf_threshold[idx], mask=None,
                        truncatedist=cfg.truncatedist[idx]
                    )
                    # # 重建使用 full grid @256
                    pred_np_field = octree2volum(
                        logit_i.detach().cpu().numpy(),
                        input_dict[f'pc_query_grid_{scale}'].cpu().numpy(),
                        tar_size=scale, thres=cfg.truncatedist[idx]
                    )
                    verts_pred, faces_pred = extract_surface_points(
                        pred_np_field, voxel_size=2.0/scale, alpha=cfg.mc_alpha[idx]
                    )
                    save_surface_as_mesh(f'{inout_dir}/pred_mesh_{scale}.ply', verts_pred, faces_pred)

                else:
                    # 第2轮：先回填到 full logits（长度=fine_total） # <<<
                    logit_full = fill_missing_as_negative(
                        pred_logits_partial=logit_i, mask=mask_i, total_voxel_num=fine_total
                    )

                    metrics = sdf_eval_metric_all_TM(
                        logit_full, 
                        input_dict[f'pc_query_sdf_{scale}'], 
                        sdf_threshold=cfg.sdf_threshold[idx],
                        mask=None,
                        truncatedist=cfg.truncatedist[idx]
                    )

                    # # 重建一定要用 partial logits + partial grid   # <<<
                    pred_np_field = octree2volum(
                        logit_i.detach().cpu().numpy(),
                        input_dict_fine['pc_query_grid_512'].cpu().numpy(), 
                        tar_size=scale, thres=cfg.truncatedist[idx]
                    )
                    verts_pred, faces_pred = extract_surface_points(
                        pred_np_field, voxel_size=2.0/scale, alpha=cfg.mc_alpha[idx]
                    )
                    save_surface_as_mesh(f'{inout_dir}/pred_mesh_{scale}.ply', verts_pred, faces_pred)

                # === structure metrics ===
                class0_metrics = {}
                class1_metrics = {}
                weighted_metrics = {}
                overall_metrics = {}

                for name, value in metrics:
                    if "cl0" in name:
                        base_name = name.replace("_cl0", "")
                        class0_metrics[base_name] = value
                    elif "cl1" in name:
                        base_name = name.replace("_cl1", "")
                        class1_metrics[base_name] = value
                    elif "wavg" in name:
                        base_name = name.replace("_wavg", "")
                        weighted_metrics[base_name] = value
                    else:
                        base_name = name
                        overall_metrics[base_name] = value

                print("\n-------------- Overall Metric ----------------")
                print("\n---------------- Overall Metric ----------------", file=f_s)
                for metric_name in ["SDF_L1", "SDF_L2"]:
                    line = f"{metric_name:<12} (Overall): {overall_metrics[metric_name]:.8f}"
                    print(line)
                    print(line, file=f_s)

                print("\n-------------- Class Metric ----------------")
                print("\n---------------- Class Metric ----------------", file=f_s)
                for metric_name in ["IoU"]:
                    line = (f"{metric_name:<12} - "
                            f"class 0: {class0_metrics[metric_name]:.4f} | "
                            f"class 1: {class1_metrics[metric_name]:.4f}")
                    print(line)
                    print(line, file=f_s)

                print("\n-------------- Weighted Average ----------------")
                print("\n---------------- Weighted Average ----------------", file=f_s)
                for metric_name in ["IoU"]:
                    line = f"{metric_name:<12} (weighted): {weighted_metrics[metric_name]:.4f}"
                    print(line)
                    print(line, file=f_s)

                ## accumulation on whole dataloader
                for name, score in metrics:
                    running_eval_metrics[f'{name}_s{scale}'] += score
    
    level_groups,_ = log_metrics(running_eval_metrics, len(dataloader))
    # =============== 写入 avg_result.txt ===============
    avg_txt_path = os.path.join(exp_name, 'avg_result.txt')
    with open(avg_txt_path, 'w') as f:
        print('================ Average over all scenes ================')
        print('================ Average over all scenes ================', file=f)
        for scale, group in level_groups.items():
            print(f"\n-- Metrics for {scale} --")
            print(f"\n-- Metrics for {scale} --", file=f)
            for k, v in sorted(group.items()):
                print(f"{k:<25}: {v:.8f}")
                print(f"{k:<25}: {v:.8f}", file=f)

                
if __name__ == "__main__":
    main()
    
    
    
