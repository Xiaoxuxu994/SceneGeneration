import torch
import open3d as o3d
import os
import numpy as np
import torchmetrics
from collections import abc

def save_pc(points, output_path, colors=None):
    """
    Save point cloud to PLY.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)

def save_colored_pc(points, mask, color, output_path):
    """
    Save masked points with specified color.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    masked_points = points[mask == 1]
    if masked_points.size == 0:
        return  # Skip saving empty files
    colors = np.tile(color, (masked_points.shape[0], 1))
    save_pc(masked_points, output_path, colors)

def save_inout_pc(inout_dir, input_dict, output):
    """
    Save full point cloud, query points, GT ON points, and predicted ON points.
    """
    os.makedirs(inout_dir, exist_ok=True)

    # add sync specialize for npu
    torch.npu.synchronize()
    # Ensure proper conversion
    coord = input_dict['coord'].detach().cpu().numpy()
    pc_query = input_dict['pc_query'].detach().cpu().numpy()
    pc_query_label = input_dict['pc_query_label'].detach().cpu().numpy()

    # Prediction probabilities
    probs = torch.sigmoid(output[0]).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.uint8)

    # Save full scene point cloud
    save_pc(coord, os.path.join(inout_dir, "pointcloud.ply"))

    # Save query points (uncolored)
    save_pc(pc_query, os.path.join(inout_dir, "pointcloud_query.ply"))

    # Save GT ON (label==1) as green
    save_colored_pc(pc_query, pc_query_label, color=[0, 1, 0], output_path=os.path.join(inout_dir, "pointcloud_query_on_gt.ply"))

    # Save predicted ON (pred==1) as red
    save_colored_pc(pc_query, preds, color=[1, 0, 0], output_path=os.path.join(inout_dir, "pointcloud_query_on_pre.ply"))


def occ_eval_metric(output_logit,label,occ_threshold):
    pred = torch.zeros_like(output_logit).cpu().detach()
    pred[output_logit>=occ_threshold] = 1
    label = label.cpu().detach()
    
    # precision      
    precision = torchmetrics.Precision(task='binary')
    precision_score = precision(pred, label)
    # recall
    recall = torchmetrics.Recall(task='binary')
    recall_score = recall(pred, label)
    # f1
    f1_metric = torchmetrics.F1Score(task='binary')
    f1_score = f1_metric(pred, label)
    # iou
    iou_metric = torchmetrics.JaccardIndex(task='binary', num_classes=2)
    iou_score = iou_metric(pred, label)     

    return precision_score, recall_score, f1_score, iou_score

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True