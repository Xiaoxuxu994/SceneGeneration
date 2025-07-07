import torch
import os
import numpy as np

def save_inout_pc(inout_dir,input_dict,output):      
    os.makedirs(inout_dir, exist_ok=True)
    x = input_dict['coord'].detach().cpu().numpy()
    pc_query = input_dict['pc_query'].detach().cpu().numpy()
    pc_query_label = input_dict['pc_query_label'].detach().cpu().numpy()
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(output[0,...]).detach().cpu().numpy()
    predictions = probabilities >= 0.5
    pcd = {
        "gt_pcd":x,
        "query_grid": pc_query,
        "query_gt": pc_query[pc_query_label==1],
        "query_pre": pc_query[predictions==1],
    }
    filename = os.path.join(inout_dir, "pointcloud.npz")
    np.save(filename, pcd)