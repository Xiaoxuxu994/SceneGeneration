import torch
import os
import numpy as np
import torchmetrics
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

def occ_eval_metric(metric_list, output_logit, label, occ_threshold):
    pred = torch.zeros_like(output_logit).cpu().detach()
    pred[output_logit>=occ_threshold] = 1
    label = label.cpu().detach()
    
    if 'precision' in metric_list:         
        precision = torchmetrics.Precision(task='binary')
        precision_score = precision(pred, label)
    if 'recall' in metric_list:
        recall = torchmetrics.Recall(task='binary')
        recall_score = recall(pred, label)
    if 'f1' in metric_list:
        f1_metric = torchmetrics.F1Score(task='binary')
        f1_score = f1_metric(pred, label)
    if 'iou' in metric_list:
        iou_metric = torchmetrics.JaccardIndex(task='binary', num_classes=2)
        iou_score = iou_metric(pred, label)     

    return precision_score, recall_score, f1_score, iou_score
           
def occ_eval_metric_(output_logit,label,occ_threshold):
    pred = torch.zeros_like(output_logit).cpu()
    pred[output_logit>=occ_threshold] = 1
    label = label.cpu()       
    precision = torchmetrics.Precision(task='binary')
    precision_score = precision(pred, label)

    recall = torchmetrics.Recall(task='binary')
    recall_score = recall(pred, label)

    f1_metric = torchmetrics.F1Score(task='binary')
    f1_score = f1_metric(pred, label)

    iou_metric = torchmetrics.JaccardIndex(task='binary', num_classes=2)
    iou_score = iou_metric(pred, label)     

    return precision_score, recall_score, f1_score, iou_score

def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# add for ctf and iou
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
    }

    # # 总体评估指标（不分层）
    # for cls in class_keys:
    #     for m in base_metrics:
    #         metrics[f"{m}_{cls}"] = 0.0
    
    # 分层指标
    if levels:
        for lvl in levels:
            metrics[f"loss_occ_s{lvl}"] = 0.0
            metrics[f"logit_mean_s{lvl}"] = 0.0
            metrics[f"logit_max_s{lvl}"] = 0.0
            metrics[f"logit_min_s{lvl}"] = 0.0
            if include_per_level:
                for cls in class_keys:
                    for m in base_metrics:
                        metrics[f"{m}_{cls}_s{lvl}"] = 0.0  # e.g., F1_cl1_s16

    return metrics

def init_running_metrics_nolvl(class_keys: list[str] = ['cl1', 'cl0', 'wavg'], base_metrics: list[str] = ['Precision', 'Recall', 'F1', 'IoU']) -> dict:
    """
    初始化 running_metrics 结构，支持逐层 loss + per-class metrics metrics

    Args:
        class_keys: 类别指标后缀，例如 ['cl1', 'cl0', 'wavg']
        base_metrics: 基本评估指标名，例如 ['Precision', 'Recall', 'F1', 'IoU']
    Returns:
        metrics: dict[str, float]
    """

    metrics = {
        'loss': 0.0,
    }

    # # 总体评估指标（不分层）
    # for cls in class_keys:
    #     for m in base_metrics:
    #         metrics[f"{m}_{cls}"] = 0.0
    
    # 分层指标
    metrics[f"loss_occ"] = 0.0
    metrics[f"logit_mean"] = 0.0
    metrics[f"logit_max"] = 0.0
    metrics[f"logit_min"] = 0.0
    for cls in class_keys:
        for m in base_metrics:
            metrics[f"{m}_{cls}"] = 0.0  # e.g., F1_cl1

    return metrics

