import torch
import open3d as o3d
import os
import numpy as np
import torchmetrics

def save_pc(x,output_path,color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        if color is not None:
             pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(output_path, pcd)

def save_inout_pc(inout_dir,input_dict,output):
        
    os.makedirs(inout_dir, exist_ok=True)
    x = input_dict['coord'].detach().cpu().numpy()
    pc_query = input_dict['pc_query'].detach().cpu().numpy()
    pc_query_label = input_dict['pc_query_label'].detach().cpu().numpy()

    # x = x[0,...].detach().cpu().numpy()
    # pc_query = pc_query[0,...].detach().cpu().numpy()
    # pc_query_label = pc_query_label[0,...].detach().cpu().numpy()

    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(output[0,...]).detach().cpu().numpy()
    predictions = probabilities >= 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.io.write_point_cloud(os.path.join(inout_dir,"pointcloud.ply"), pcd)

    pcd_query = o3d.geometry.PointCloud()
    pcd_query.points = o3d.utility.Vector3dVector(pc_query)
    o3d.io.write_point_cloud(os.path.join(inout_dir,"pointcloud_query.ply"), pcd_query)

    
    pc_query_on = pc_query[pc_query_label==1]
    pcd_query_on = o3d.geometry.PointCloud()
    pcd_query_on.points = o3d.utility.Vector3dVector(pc_query_on)
    colors_on = np.tile([0,1,0],(pc_query_on.shape[0],1)) ## green 
    pcd_query_on.colors = o3d.utility.Vector3dVector(colors_on)
    o3d.io.write_point_cloud(os.path.join(inout_dir,"pointcloud_query_on_gt.ply"), pcd_query_on)
    
    pc_query_on = pc_query[predictions==1]
    pcd_query_on = o3d.geometry.PointCloud()
    pcd_query_on.points = o3d.utility.Vector3dVector(pc_query_on)
    colors_on = np.tile([1,0,0],(pc_query_on.shape[0],1)) ## red
    pcd_query_on.colors = o3d.utility.Vector3dVector(colors_on)
    o3d.io.write_point_cloud(os.path.join(inout_dir,"pointcloud_query_on_pre.ply"), pcd_query_on)

def save_inout_pc_v2(inout_dir,input_dict,output,pc_query_in=None,pc_query_label_in=None,prefix='0.05'):
        
    os.makedirs(inout_dir, exist_ok=True)
    x = input_dict['coord'].detach().cpu().numpy()
    pc_query = input_dict[f'pc_query_{prefix}'].detach().cpu().numpy()
    pc_query_label = input_dict[f'pc_query_label_{prefix}'].detach().cpu().numpy()

    if pc_query_in is not None:
        pc_query_in = pc_query_in.detach().cpu().numpy()
        save_pc(pc_query_in,os.path.join(inout_dir,f"pointcloud_query_in_{prefix}.ply"),color=None)

        pc_query_label_in = pc_query_label_in.detach().cpu().numpy()
        pc_query_in_on = pc_query_in[pc_query_label_in==1]
        colors_on = np.tile([0,1,0],(pc_query_in_on.shape[0],1)) ## green 
        save_pc(pc_query_in,os.path.join(inout_dir,f"pointcloud_query_in_{prefix}.ply"),color=colors_on)

    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(output[0,...]).detach().cpu().numpy()
    predictions = probabilities >= 0.5

    # save org input
    save_pc(x,os.path.join(inout_dir,"pointcloud.ply"),color=None)
    save_pc(pc_query,os.path.join(inout_dir,f"pointcloud_query_{prefix}.ply"),color=None)
    
    pc_query_on = pc_query[pc_query_label==1]
    colors_on = np.tile([0,1,0],(pc_query_on.shape[0],1)) ## green 
    save_pc(pc_query_on,os.path.join(inout_dir,f"pointcloud_query_on_gt_{prefix}.ply"),color=colors_on)
    
    if pc_query_in is not None:
        pc_query_on = pc_query_in[predictions==1]
        colors_on = np.tile([1,0,0],(pc_query_on.shape[0],1)) ## red
        save_pc(pc_query_on,os.path.join(inout_dir,f"pointcloud_query_on_pre_{prefix}.ply"),color=colors_on)             
    else:
        pc_query_on = pc_query[predictions==1]
        colors_on = np.tile([1,0,0],(pc_query_on.shape[0],1)) ## red
        save_pc(pc_query_on,os.path.join(inout_dir,f"pointcloud_query_on_pre_{prefix}.ply"),color=colors_on)

def save_inout_pc_input(inout_dir, input_dict):
    os.makedirs(inout_dir, exist_ok=True)
    x = input_dict['coord'].detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.io.write_point_cloud(os.path.join(inout_dir,"pointcloud.ply"), pcd)

def save_inout_pc_ctf(inout_dir, input_dict, logit, mask, scale, occ_threshold=0.5):
        
    os.makedirs(inout_dir, exist_ok=True)

    pc_query = input_dict[f'pc_query_{scale}'].detach().cpu().numpy()

    if scale == 128:
        pc_query_label = input_dict[f'pc_query_label_{scale}'].detach().cpu().numpy()
    else:
        pc_query_label = input_dict[f'pc_query_label_fine_prop_{scale}'].detach().cpu().numpy()

    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(logit).detach().cpu().numpy()
    predictions = probabilities >= occ_threshold

    pcd_query = o3d.geometry.PointCloud()
    if mask is not None:
        pcd_query.points = o3d.utility.Vector3dVector(pc_query[mask.detach().cpu().numpy()])
    else:
        pcd_query.points = o3d.utility.Vector3dVector(pc_query)
    o3d.io.write_point_cloud(os.path.join(inout_dir,f"pointcloud_query_{scale}.ply"), pcd_query)

    pc_query_on = pc_query[pc_query_label==1]
    pcd_query_on = o3d.geometry.PointCloud()
    pcd_query_on.points = o3d.utility.Vector3dVector(pc_query_on)
    colors_on = np.tile([0,1,0],(pc_query_on.shape[0],1)) ## green 
    pcd_query_on.colors = o3d.utility.Vector3dVector(colors_on)
    o3d.io.write_point_cloud(os.path.join(inout_dir,f"pointcloud_query_on_gt_{scale}.ply"), pcd_query_on)
    
    pc_query_on = pc_query[predictions==1]
    pcd_query_on = o3d.geometry.PointCloud()
    pcd_query_on.points = o3d.utility.Vector3dVector(pc_query_on)
    colors_on = np.tile([1,0,0],(pc_query_on.shape[0],1)) ## red
    pcd_query_on.colors = o3d.utility.Vector3dVector(colors_on)
    o3d.io.write_point_cloud(os.path.join(inout_dir,f"pointcloud_query_on_pre_{scale}.ply"), pcd_query_on)


def save_surface_as_mesh(filepath, verts, faces):
    """
    将提取到的表面网格保存为 .ply 文件
    Args:
        filepath: str, 例如 "mesh.ply"
        verts: (N, 3) numpy array
        faces: (M, 3) numpy array of triangle indices
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.75, 0.75, 0.75])
    o3d.io.write_triangle_mesh(filepath, mesh, write_ascii=True)
    
def occ_eval_metric(metric_list,output_logit,label,occ_threshold):
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
