# %%
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch._C._jit_set_profiling_mode(False)
from torch.utils.data import DataLoader
from tqdm import tqdm

from opensora_local.utils.ckpt_utils import load_checkpoint_ckpt
from opensora_local.utils.config_utils import parse_configs

## src code
from src.registry import MODELS, DATASETS, build_module
from src.utils.savepcd import save_inout_pc
from src.utils.metric import occ_eval_metric_all, init_running_metrics_nolvl
from src.dataset.dataset_build_scannet_v2 import DatasetForScanNetV2

def main():
    # args loading and write to experiment dir
    cfg = parse_configs(training=True)
    print("====== cfg: ",cfg)
    device = "npu:0"
    dtype = torch.float32
    
    # build DatasetForScanNetV2
    val_dataset = build_module(cfg.val_dataset, DATASETS)
    dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    num_val = len(dataloader)

    # exp_output_dir
    ckpt_path = os.path.join(cfg.root, cfg.ckpt_name)
    exp_name = os.path.join(cfg.outputs, cfg.ckpt_name.split('.')[0], cfg.prefix)
    os.makedirs(exp_name, mode = 0o777, exist_ok = True)
    print("====== data saving to : ", exp_name)

    # model ckpt load
    print("====== cfg.model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint_ckpt(model, ckpt_path, device)
    model.eval()

    running_eval_metrics = init_running_metrics_nolvl(class_keys=['cl1','cl0','wavg'], base_metrics=['Precision','Recall','F1','IoU'])
    for input_dict in tqdm(dataloader):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key][0].to(device)

        with torch.no_grad():
            # output_list, target_list, mask_list = model.forward_gt(input_dict,return_mask=True)
            output = model.forward(input_dict)

        # === save and print ===
        sub_folder = input_dict['scene_id'][0]
        inout_dir = os.path.join(exp_name, 'scene', sub_folder)
        os.makedirs(inout_dir, exist_ok=True)
        # save_inout_pc_input(inout_dir, input_dict)

        result_txt_path = os.path.join(inout_dir, 'result.txt')
        # ======== 计算 OCC 指标 =========
        print(f"\n================ Scene: {sub_folder} =================")
        with open(result_txt_path, 'w') as f_s:
            print('================ Per-scene Evaluation =================', file=f_s)
            print(f"Scene: {sub_folder}")
            print(f"Scene: {sub_folder}", file=f_s)

            # save_inout_pc_ctf(inout_dir, input_dict, logit, ocmask, scale, c_threshold=cfg.occ_threshold)
            metrics = occ_eval_metric_all(output['logits'], input_dict[f'pc_query_label'], occ_threshold=cfg.occ_threshold)

            # === structure metrics ===
            class0_metrics = {}
            class1_metrics = {}
            weighted_metrics = {}

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
                running_eval_metrics[f'{name}'] += score
    
    # =============== 写入 avg_result.txt ===============
    avg_txt_path = os.path.join(exp_name, 'avg_result.txt')
    with open(avg_txt_path, 'w') as f:
        print('================ Average over all scenes ================')
        print('================ Average over all scenes ================', file=f)
        for k, v in sorted(running_eval_metrics.items()):
            print(f"{k:<25}: {v:.4f}, {v/num_val:.4f}")
            print(f"{k:<25}: {v:.4f}, {v/num_val:.4f}", file=f)

if __name__ == "__main__":
    main()