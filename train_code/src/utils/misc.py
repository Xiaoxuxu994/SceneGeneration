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

