import os
import torch
import torch_npu
from torch.utils.data import DataLoader
torch._C._jit_set_profiling_mode(False)
from tqdm import tqdm

## src code
from src.registry import MODELS, build_module
from src.models.geometryvae import *
from src.dataset.dataset_build_scannet_v2 import DatasetForScanNetV2Test
from src.utils.config_utils import parse_configs
from src.utils.ckpt_utils import load_checkpoint
from src.utils.misc import save_inout_pc
from src.utils.metrics import OccMetricEvaluator

def evaluate_one_sample(input_dict, model, base_dir, device, evaluator):
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key][0].to(device)

    with torch.no_grad():
        outputs = model(input_dict)

    pc_query_label = input_dict['pc_query_label'].unsqueeze(0)
    output_logit = outputs['logits']

    precision_score, recall_score, f1_score, iou_score = evaluator(output_logit, pc_query_label)

    sub_folder = input_dict['scene_id'][0]
    inout_dir = os.path.join(base_dir, sub_folder)
    os.makedirs(inout_dir, exist_ok=True)

    save_inout_pc(inout_dir, input_dict, output_logit)

    with open(os.path.join(inout_dir, 'result.txt'), 'w') as f_s:
        for name, score in zip(['iou_score', 'precision', 'recall', 'f1_score'],
                               [iou_score.mean(), precision_score.mean(), recall_score.mean(), f1_score.mean()]):
            print(f"Occ {name}: {score.item()}")
            print(f"Occ {name}: {score.item()}", file=f_s)

    scores = {
        "iou": iou_score.mean().item(),
        "precision": precision_score.mean().item(),
        "recall": recall_score.mean().item(),
        "f1": f1_score.mean().item()
    }

    del input_dict, output_logit, pc_query_label
    torch_npu.npu.empty_cache()

    return scores

def evaluate_one_loop(model, dataloader, base_dir, device, evaluator):
    cumulative_scores = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    count = 0

    for input_dict in (tbar := tqdm(dataloader)):
        scores = evaluate_one_sample(input_dict, model, base_dir, device, evaluator)
        for k in cumulative_scores:
            cumulative_scores[k] += scores[k]
        count += 1

    avg_scores = {k: v / count for k, v in cumulative_scores.items()}
    print(f"======= Average over {count} samples =======")
    for k, v in avg_scores.items():
        print(f"Mean Occ {k}: {v}")

    with open(os.path.join(base_dir, 'avg_result.txt'), 'w') as f:
        for k, v in avg_scores.items():
            print(f"Mean Occ {k}: {v}", file=f)

    return avg_scores
        
def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs()
    print("=" * 30 + " cfg " + "=" * 30)
    for name in dir(cfg):
        if not name.startswith("__"):
            print(f"{name} = {getattr(cfg, name)}")
    print("=" * 60)
    device = torch.device('npu:0')
    dtype = torch.float32

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet
    dataset = DatasetForScanNetV2Test(
        data_root=cfg.data_root,
        data_list_name=cfg.val_data_list_name,
        query_sample_rate=1.0,
        max_query_number=1000000,
        normalize_coord = cfg.normalize_coord,
        Gridsample_mode = 'train',
        shufflepoint = False,
        grid_size = cfg.grid_size
    )

    dataloader = DataLoader(dataset,batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    # ======================================================
    # 3. exp output dir
    # ======================================================
    ckpt_path = os.path.join(cfg.root,cfg.ckpt_name,'ckpt484.pth')
    exp_name = os.path.join(cfg.outputs, cfg.ckpt_name, cfg.prefix)
    os.makedirs(exp_name, mode = 0o777, exist_ok = True)
    print("====== data saving to : ", exp_name)

    # ======================================================
    # 4. build model
    # ======================================================
    print("====== model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint(model, ckpt_path, strict=True)
    model.eval()

    # =======================================================
    # 5. Eval Loop
    # =======================================================
    evaluator = OccMetricEvaluator(occ_threshold=cfg.occ_threshold)
    n_iter = 5
    iou, precision, recall, f1 = [], [], [], []
    for i in range(n_iter):
        print(f'---------- {i+1}/{n_iter} loop ----------')
        base_dir = os.path.join(exp_name, f"{i:02d}")
        avg_scores = evaluate_one_loop(model, dataloader, base_dir, device, evaluator)

        iou.append(avg_scores["iou"])
        precision.append(avg_scores["precision"])
        recall.append(avg_scores["recall"])
        f1.append(avg_scores["f1"])

    print(f'================ Final Avg over {n_iter} iterations ================')

    final_scores = {
        "iou": sum(iou)/n_iter,
        "precision": sum(precision)/n_iter,
        "recall": sum(recall)/n_iter,
        "f1": sum(f1)/n_iter
    }
    for k, v in final_scores.items():
        print(f"Mean Occ {k}: {v}")

    with open(os.path.join(exp_name, f'avg_result_{n_iter}.txt'), 'w') as f:
        for k, v in final_scores.items():
            print(f"Mean Occ {k}: {v}", file=f)

if __name__ == "__main__":
    main()
    
    
    
