"""
We assume that all the datasets share the same folder structure as following, 
|-scene0000001
    |-viewinfo.npz (camera infos)
    |-images
    |   |-prefix00001.png/JPG
    |   |-prefix00002.png/JPG
    |   |-...
    |-depths
    |   |-prefix00001.png
    |   |-prefix00002.png
    |   |-...
    |-depth_confs
    |   |-prefix00001.npy
    |   |-prefix00002.npy
    |   |-...
    |-matchings
        |-collection00001.npy
        |-collection00002.npy
        |-...
"""

import torch

from torch.utils.data import Dataset, DataLoader


class FF3D_Dataset(Dataset):
    def __init__(self, ):
        
        self.meta_data_files = []
        """
        For Reliev3R, we formulate data as video frame segments. 
        Each segment is treated as a meta data sample, which can be subsampled to change the actual frame length input to model training.
        Each meta data sample is defined as, 
        dict(
            image_files: list of file paths to read original images; 
            ps_depth_files: list of file paths to read pseudo depths; 
            gt_depth_files (optional): list of file paths to read ground truth depths, optional when performing supervised learning.
            matching_file: a file path to read matching of image_files; 
            camera_extrs: list of camera extrinsics, numpy arrays; 
            camera_intrs: list of camera intrinsics, numpy arrays; 
        )
        Each meta data sample is storejsond in a . file, which is an element of self.meta_data_files. 
        """
        
        
        
    
    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
    
    