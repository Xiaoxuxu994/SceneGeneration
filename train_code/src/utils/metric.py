import torch
import os
import numpy as np
import torchmetrics

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