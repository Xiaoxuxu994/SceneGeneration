import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from skimage.measure import marching_cubes
# from pytorch3d.loss import chamfer_distance
# import ot
import numpy as np
from src.utils.misc import save_surface_as_mesh

def occ_eval_metric_all(output_logit, label, occ_threshold=0.5):
    """
    Compute per-class (0 & 1) and weighted metrics:
    - IoU, Precision, Recall, F1 Score

    Returns:
        metrics (list of (name, value))
    """
    # Apply sigmoid and binarize
    prob = torch.sigmoid(output_logit)
    pred = (prob >= occ_threshold).int()
    label = label.int()

    # Flatten to numpy arrays
    pred_np = pred.cpu().numpy().flatten()
    label_np = label.cpu().numpy().flatten()

    metrics = []

    # Per-class metrics (class 0 and class 1)
    # Compute all per-class metrics using average=None, labels=[0,1]
    ious = jaccard_score(label_np, pred_np, average=None, labels=[0, 1])
    precisions = precision_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)
    recalls = recall_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)
    f1s = f1_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)

    for c in [0, 1]:
        metrics.append((f"IoU_cl{c}", ious[c]))
        metrics.append((f"Precision_cl{c}", precisions[c]))
        metrics.append((f"Recall_cl{c}", recalls[c]))
        metrics.append((f"F1_cl{c}", f1s[c]))

    # Weighted average metrics over both classes
    metrics.append(("IoU_wavg", jaccard_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("Precision_wavg", precision_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("Recall_wavg", recall_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("F1_wavg", f1_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))

    return metrics

def sdf_occ_eval_metric_all(output_logit, label, sdf_threshold=0.5, mask=None):

    """
    Args:
        output_logit: torch.Tensor, raw SDF/UDF prediction (no sigmoid applied)
        label: torch.Tensor, ground truth sdf/udf values
        sdf_threshold: float, threshold to binarize (e.g. 0.5 means outside)
        mask: optional, binary mask to indicate valid regions
    Returns:
        metrics: list of (name, value)
    """
    # 1. Binary prediction: abs(sdf) > threshold is considered free space (0), else occupied (1)
    pred = (output_logit <= sdf_threshold).int()
    label = (label <= sdf_threshold).int()

    if mask is not None:
        mask = mask.bool()
        pred = pred[mask]
        label = label[mask]

    # Flatten to numpy arrays
    pred_np = pred.cpu().numpy().flatten()
    label_np = label.cpu().numpy().flatten()

    metrics = []

    # Per-class metrics (class 0 and class 1)
    # Compute all per-class metrics using average=None, labels=[0,1]
    ious = jaccard_score(label_np, pred_np, average=None, labels=[0, 1])
    precisions = precision_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)
    recalls = recall_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)
    f1s = f1_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)

    for c in [0, 1]:
        metrics.append((f"IoU_cl{c}", ious[c]))
        metrics.append((f"Precision_cl{c}", precisions[c]))
        metrics.append((f"Recall_cl{c}", recalls[c]))
        metrics.append((f"F1_cl{c}", f1s[c]))

    # Weighted average metrics over both classes
    metrics.append(("IoU_wavg", jaccard_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("Precision_wavg", precision_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("Recall_wavg", recall_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("F1_wavg", f1_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))

    return metrics

def extract_surface_points(udf_field, voxel_size=2.0/128, alpha = 0.5):
    """
    从 UDF 场中提取表面点（UDF ≈ 0）使用 marching cubes.
    Args:
        udf_field: numpy array of shape (D, H, W)
        threshold: 提取的等值面 UDF 值
        spacing: 体素间距
    Returns:
        verts: (N, 3) 表面点坐标
        faces: (M, 3) 三角面片索引
    """
    verts, faces, _, _ = marching_cubes(udf_field, 
                                        level=voxel_size*alpha, 
                                        spacing=(voxel_size, voxel_size, voxel_size))
    # shifts = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= 1.0

    return verts,faces


# def compute_chamfer(pred_points, gt_points):
#     """
#     计算 Chamfer Distance
#     Args:
#         pred_points: numpy (N, 3)
#         gt_points: numpy (M, 3)
#     Returns:
#         chamfer distance (float)
#     """
#     pred = torch.from_numpy(pred_points).float().unsqueeze(0).cuda()
#     gt = torch.from_numpy(gt_points).float().unsqueeze(0).cuda()
#     cd, _ = chamfer_distance(pred, gt)
#     return cd.item()

# def compute_emd(pred_points, gt_points, eps=0.005, iters=50):
#     """
#     计算 Earth Mover's Distance（使用 PyTorch3D 的 approx EMD）
#     Args:
#         pred_points: numpy (N, 3)
#         gt_points: numpy (M, 3)
#     Returns:
#         emd distance (float)
#     """
#     assert pred_points.shape[0] == gt_points.shape[0], "EMD requires equal number of points"
#     pred = torch.from_numpy(pred_points).float().unsqueeze(0).cuda()
#     gt = torch.from_numpy(gt_points).float().unsqueeze(0).cuda()
#     # emd = emd_loss(pred, gt, eps=eps, iters=iters)
#     return emd.item()

# def compute_emd(pred_points, gt_points):
#     """
#     使用 POT 库计算 EMD，适合评估用，不可微
#     Args:
#         pred_points: numpy (N, 3)
#         gt_points: numpy (M, 3)
#     Returns:
#         emd distance (float)
#     """
#     n = min(len(pred_points), len(gt_points))
#     if n == 0:
#         return float('inf')

#     # Uniform weights
#     a = np.ones((n,)) / n
#     b = np.ones((n,)) / n

#     M = ot.dist(pred_points[:n], gt_points[:n])  # 欧几里得距离矩阵
#     emd = ot.emd2(a, b, M)
#     return emd

def udf_l1_l2_metric(output_logit, label, mask=None,truncatedist=None):
    """
    计算 L1 和 L2（RMSE）误差
    """
    if mask is not None:
        output_logit = output_logit[mask]
        label = label[mask]
    if truncatedist:
        label = torch.clip(label,0.0,truncatedist)
        output_logit = torch.clip(output_logit,0.0,truncatedist)
    abs_diff = torch.abs(output_logit - label)
    sqr_diff = (output_logit - label) ** 2
    l1 = abs_diff.mean().item()
    l2 = torch.sqrt(sqr_diff.mean()).item()
    return [('SDF_L1', l1), ('SDF_L2', l2)]

def sdf_eval_metric_all(output_logit, label, sdf_threshold=0.5, mask=None,
                        truncatedist=None,
                        compute_surface_metrics=False, spacing=1.0,
                        voxel_res = 128,
                        inout_dir=None,prefix=None):
    
    """
    Args:
        output_logit: torch.Tensor, raw SDF/UDF prediction (no sigmoid applied)
        label: torch.Tensor, ground truth sdf/udf values
        sdf_threshold: float, threshold to binarize (e.g. 0.5 means outside)
        mask: optional, binary mask to indicate valid regions
    Returns:
        metrics: list of (name, value)
    """  
    #### 1. binary occ evaluation
    # Binary prediction: abs(sdf) > threshold is considered free space (0), else occupied (1)
    pred = (output_logit <= sdf_threshold).int()
    label_bin = (label <= sdf_threshold).int()
    
    if mask is not None:
        mask = mask.bool()
        pred = pred[mask]
        label_bin = label_bin[mask]

    # Flatten to numpy arrays
    pred_np = pred.cpu().numpy().flatten()
    label_np = label_bin.cpu().numpy().flatten()

    metrics = []
    # Per-class metrics (class 0 and class 1)
    # Compute all per-class metrics using average=None, labels=[0,1]
    ious = jaccard_score(label_np, pred_np, average=None, labels=[0, 1])
    precisions = precision_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)
    recalls = recall_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)
    f1s = f1_score(label_np, pred_np, average=None, labels=[0, 1], zero_division=0)

    for c in [0, 1]:
        metrics.append((f"IoU_cl{c}", ious[c]))
        metrics.append((f"Precision_cl{c}", precisions[c]))
        metrics.append((f"Recall_cl{c}", recalls[c]))
        metrics.append((f"F1_cl{c}", f1s[c]))

    # Weighted average metrics over both classes
    metrics.append(("IoU_wavg", jaccard_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("Precision_wavg", precision_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("Recall_wavg", recall_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))
    metrics.append(("F1_wavg", f1_score(label_np, pred_np, labels=[0, 1], average='weighted', zero_division=0)))

    ### 2. udf l1&l2计算
    metrics.extend(udf_l1_l2_metric(output_logit, label, mask, truncatedist))

    ### 3. 表面距离指标
    if compute_surface_metrics:
        grid_size = [voxel_res,voxel_res,voxel_res]
        try:
            pred_np_field = output_logit.view(*grid_size).detach().cpu().numpy()
            label_np_field = label.view(*grid_size).detach().cpu().numpy()
            verts_pred, faces_pred = extract_surface_points(pred_np_field, voxel_size=2.0/voxel_res, alpha = 1.0)
            verts_label, faces_label = extract_surface_points(label_np_field, voxel_size=2.0/voxel_res, alpha = 1.0)
            # if len(verts_pred) > 0 and len(verts_label) > 0:
                # cd = compute_chamfer(verts_pred, verts_label)
                # emd = compute_emd(verts_pred, verts_label)
                # metrics.append(('ChamferDist', cd))
                # metrics.append(('EMD', emd))
            if inout_dir is not None:
                save_surface_as_mesh(f'{inout_dir}/pred_mesh_{prefix}.ply', verts_pred, faces_pred)
                save_surface_as_mesh(f'{inout_dir}/gt_mesh_{prefix}.ply', verts_label, faces_label)
        except Exception as e:
            print(f"[warning] Surface extraction failed: {e}")

    return metrics

def init_running_metrics(levels: list[int] = None,
                         class_keys: list[str] = ['cl1', 'cl0', 'wavg'],
                         base_metrics: list[str] = ['Precision', 'Recall', 'F1', 'IoU'],
                         include_per_level: bool = True) -> dict:
    """
    初始化 running_metrics 结构，支持逐层 loss + per-class metrics + per-level metrics

    Args:
        levels: 层级列表，如 [16, 32, 64, 128]
        class_keys: 类别指标后缀，例如 ['cl1', 'cl0', 'wavg']
        base_metrics: 基本评估指标名，例如 ['Precision', 'Recall', 'F1', 'IoU']
        include_per_level: 是否为每层记录所有指标
    Returns:
        metrics: dict[str, float]
    """

    metrics = {
        'loss': 0.0,
        'loss_occ': 0.0,
        'loss_kl': 0.0,
        'loss_sdf': 0.0,
        'grad_norm':0.0,
    }

    # # 总体评估指标（不分层）
    # for cls in class_keys:
    #     for m in base_metrics:
    #         metrics[f"{m}_{cls}"] = 0.0
    
    # 分层指标
    if levels:
        for lvl in levels:
            metrics[f"loss_occ_s{lvl}"] = 0.0
            metrics[f"loss_sdf_s{lvl}"] = 0.0
            metrics[f"logit_mean_s{lvl}"] = 0.0
            metrics[f"logit_max_s{lvl}"] = 0.0
            metrics[f"logit_min_s{lvl}"] = 0.0
            metrics[f"SDF_L1_s{lvl}"] = 0.0
            metrics[f"SDF_L2_s{lvl}"] = 0.0
            # metrics[f"ChamferDist_s{lvl}"] = 0.0
            # metrics[f"EMD_s{lvl}"] = 0.0
            if include_per_level:
                for cls in class_keys:
                    for m in base_metrics:
                        metrics[f"{m}_{cls}_s{lvl}"] = 0.0  # e.g., F1_cl1_s16

    return metrics
