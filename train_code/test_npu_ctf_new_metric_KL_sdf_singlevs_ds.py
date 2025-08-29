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
from src.utils.misc import save_inout_pc_input
from src.utils.metric import sdf_eval_metric_all,init_running_metrics

def log_metrics(metrics, step_count):
    avg_metrics = {k: v / step_count for k, v in metrics.items()}

    default_keys = [
        'SDF_L1','SDF_L2',
        'IoU_cl1', 'IoU_cl0', 'IoU_wavg',
        'Precision_cl1', 'Precision_cl0', 'Precision_wavg',
        'Recall_cl1', 'Recall_cl0', 'Recall_wavg',
        'F1_cl1', 'F1_cl0', 'F1_wavg']
    level_groups = {}

    for k, v in avg_metrics.items():
        # 仅保留前缀匹配的项
        if not any(k.startswith(prefix) for prefix in default_keys):
            continue

        # 分类为多层级还是默认
        if '_s' in k:
            *prefix, scale = k.split('_s')
            name = '_s'.join(prefix)
            scale = f's{scale}'
            if scale not in level_groups:
                level_groups[scale] = {}
            level_groups[scale][name] = v
    return level_groups

def fill_missing_as_negative(pred_logits_partial, mask, total_voxel_num):
    """
    Args:
        pred_logits_partial: (M,) float tensor, logits on selected query points
        mask: (N,) bool tensor, which voxels in full set were selected (M==sum(mask))
        total_voxel_num: int, N

    Returns:
        full_logits: (N,) float tensor, where unselected ones are set to -inf
    """
    full_logits = torch.full((total_voxel_num,), float('-inf'), device=pred_logits_partial.device)
    full_logits[mask] = pred_logits_partial
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

        with torch.no_grad():
            kl, output_list, target_list = model.forward(input_dict,vs=scale_level[-1])

        # === save and print ===
        sub_folder = input_dict['scene_id'][0]
        inout_dir = os.path.join(exp_name, sub_folder)
        os.makedirs(inout_dir, exist_ok=True)
        save_inout_pc_input(inout_dir, input_dict)

        result_txt_path = os.path.join(inout_dir, 'result.txt')
        # ======== 计算 OCC 指标 =========
        print(f"\n================ Scene: {sub_folder} =================")
        with open(result_txt_path, 'w') as f_s:
            print('================ Per-scene Evaluation =================', file=f_s)
            print(f"Scene: {sub_folder}")
            print(f"Scene: {sub_folder}", file=f_s)

            ### ---- train vs 128 only ------
            for idx,logit in enumerate(output_list):
                scale = scale_level[idx]

                print(f"\n***************[vs resolution: {scale}]***************")
                print(f"\n***************[vs resolution: {scale}]***************", file=f_s)       

                if scale == vs_f:
                    metrics = sdf_eval_metric_all(logit, input_dict[f'pc_query_sdf_{scale}'], 
                                                grid = input_dict[f'pc_query_grid_{scale}'].cpu().numpy(), octree_save=True,
                                                sdf_threshold=cfg.sdf_threshold,mask=None,
                                                truncatedist = cfg.truncatedist,
                                                compute_surface_metrics=True, mc_alpha=cfg.mc_alpha,
                                                voxel_res = scale,
                                                inout_dir=inout_dir,prefix=f'{scale}')
                else:
                    metrics = sdf_eval_metric_all(logit, input_dict[f'pc_query_label_fine_prop_{scale}'], sdf_threshold=cfg.sdf_threshold) ## TODO

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
                for metric_name in ["IoU", "Precision", "Recall", "F1"]:
                    line = (f"{metric_name:<12} - "
                            f"class 0: {class0_metrics[metric_name]:.4f} | "
                            f"class 1: {class1_metrics[metric_name]:.4f}")
                    print(line)
                    print(line, file=f_s)

                print("\n-------------- Weighted Average ----------------")
                print("\n---------------- Weighted Average ----------------", file=f_s)
                for metric_name in ["IoU", "Precision", "Recall", "F1"]:
                    line = f"{metric_name:<12} (weighted): {weighted_metrics[metric_name]:.4f}"
                    print(line)
                    print(line, file=f_s)

                ## accumulation on whole dataloader
                for name, score in metrics:
                    running_eval_metrics[f'{name}_s{scale}'] += score
    
    level_groups = log_metrics(running_eval_metrics, len(dataloader))
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
    
    
    
