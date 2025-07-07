import os
import torch
from torch.utils.data import Dataset
from src.dataset.transform import *
import torch.distributed as dist
from src.registry import DATASETS
import numpy as np
from util import sample_farthest_points
import time

@DATASETS.register_module()
class DatasetForScanNetV2(Dataset):
    def __init__(
        self,
        data_root: str,
        data_list_name: str='data_list.txt',
        query_sample_rate: float=1.0,
        max_query_number: int=100000,
        Gridsample_mode: str='train',
        grid_size: float=0.05,
    ):

        """
        Args:
            data_root: Path to root directory containing .pt data
            data_list_name: File containing list of sample paths (relative or absolute)
            query_sample_rate: Sampling ratio for pc_query
            max_query_number: Max number of points to sample from each label class
            Gridsample_mode: 'train' or 'val'
            grid_size: voxel size for GridSample
        """

        self.data_root = data_root
        with open(data_list_name, 'r') as f:
            self.samples = [os.path.join(self.data_root, p.strip()) for p in f.readlines()]
        self.data_list_name = data_list_name
        
        self.query_sample_rate = query_sample_rate
        self.max_query_number = max_query_number
        
        # data aug 
        self.T_GridSample = GridSample(
                        grid_size = grid_size,
                        hash_type="fnv",
                        mode = Gridsample_mode,
                        keys=("coord", "color","normal"),
                        return_grid_coord=True
                        )
        self.T_ToTensor = ToTensor()

    def sample_pc_query(self, points, labels):

        points_label_0 = points[labels == 0]
        points_label_1 = points[labels == 1]

        # deal with label 0
        num_points_label_0 = len(points_label_0)
        indices_label_0 = torch.randperm(num_points_label_0)[:min(self.max_query_number,int(num_points_label_0 * self.query_sample_rate))]
        points_label_0_down = points_label_0[indices_label_0]
        labels_label_0_down = torch.zeros(len(indices_label_0), dtype=torch.float32)

        # deal with label 1
        num_points_label_1 = len(points_label_1)
        indices_label_1 = torch.randperm(num_points_label_1)[:min(self.max_query_number,int(num_points_label_1 * self.query_sample_rate))]
        points_label_1_down = points_label_1[indices_label_1]
        labels_label_1_down = torch.ones(len(indices_label_1), dtype=torch.float32)


        # combine & shuffle
        points_combined = torch.cat((points_label_0_down, points_label_1_down), dim=0)
        labels_combined = torch.cat((labels_label_0_down, labels_label_1_down), dim=0)


        indices_combined = torch.randperm(len(points_combined))
        points_combined = points_combined[indices_combined]
        labels_combined = labels_combined[indices_combined]

        return points_combined, labels_combined


    def prepare_pointcloud_offline(self, pc_query,pc_query_label):

        # subsample rate
        if self.query_sample_rate != 1.0:
            pc_query,pc_query_label = self.sample_pc_query(pc_query,pc_query_label)     
            return pc_query, pc_query_label
        else:
            return pc_query, pc_query_label

                                
    def prepare_train_data(self, input_dict):

        input_dict = self.T_GridSample(input_dict)
        input_dict = self.T_ToTensor(input_dict)

        coord = input_dict["coord"]
        grid_coord = input_dict["grid_coord"]
        color = input_dict["color"]
        scene_id = input_dict["scene_id"]
        pc_query = input_dict['pc_query_0.05']
        pc_query_label = input_dict['pc_query_label_0.05']

        # downsample
        pc_query, pc_query_label = self.prepare_pointcloud_offline(pc_query,pc_query_label)

        result_dict = {
            'coord': coord,
            'grid_coord': grid_coord,
            'color': color,
            'scene_id': scene_id,
            'offset': torch.tensor([coord.shape[0]]),
            'feat': coord,
            'pc_query': pc_query,
            'pc_query_label': pc_query_label,
            'pc_query_offset': torch.tensor([pc_query.shape[0]]),
            'data_type': 0,
        }            
        return result_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)

# offline fps dataset fps/(pcd x epoch)
@DATASETS.register_module()
class DatasetForScanNetV3(Dataset):
    def __init__(
        self,
        data_root: str,
        fps_root: str,
        data_list_name: str='data_list.txt',
        query_sample_rate: float=1.0,
        max_query_number: int=100000,
        Gridsample_mode: str='train',
        grid_size: float=0.05,
    ):

        """
        Args:
            data_root: Path to root directory containing .pth data
            fps_root: Path to root directory containing .pth index (epoch)
            data_list_name: File containing list of sample paths (relative or absolute)
            query_sample_rate: Sampling ratio for pc_query
            max_query_number: Max number of points to sample from each label class
            Gridsample_mode: 'train' or 'val'
            grid_size: voxel size for GridSample
        """

        self.data_root = data_root
        self.fps_root = fps_root
        epoch_list = sorted(os.listdir(self.fps_root)) # epoch 000000, 000001, ..., 999999 <>= 1000000
        self.epoch_num = len(epoch_list)
        self.samples = []
        self.indexset = [[] for _ in range(self.epoch_num)]
        with open(data_list_name, 'r') as f:
            for p in f.readlines():
                self.samples.append(os.path.join(self.data_root, p.strip()))
                for epoch in range(self.epoch_num):
                    self.indexset[epoch].append(os.path.join(self.fps_root, epoch_list[epoch], p.strip()))

        self.query_sample_rate = query_sample_rate
        self.max_query_number = max_query_number
        self.current_epoch = 0
        # data aug 
        self.T_GridSample = GridSample(
                        grid_size = grid_size,
                        hash_type="fnv",
                        mode = Gridsample_mode,
                        keys=("coord", "color","normal"),
                        return_grid_coord=True
                        )
        self.T_ToTensor = ToTensor()

    def sample_pc_query(self, points, labels):

        points_label_0 = points[labels == 0]
        points_label_1 = points[labels == 1]

        # deal with label 0
        num_points_label_0 = len(points_label_0)
        indices_label_0 = torch.randperm(num_points_label_0)[:min(self.max_query_number,int(num_points_label_0 * self.query_sample_rate))]
        points_label_0_down = points_label_0[indices_label_0]
        labels_label_0_down = torch.zeros(len(indices_label_0), dtype=torch.float32)

        # deal with label 1
        num_points_label_1 = len(points_label_1)
        indices_label_1 = torch.randperm(num_points_label_1)[:min(self.max_query_number,int(num_points_label_1 * self.query_sample_rate))]
        points_label_1_down = points_label_1[indices_label_1]
        labels_label_1_down = torch.ones(len(indices_label_1), dtype=torch.float32)


        # combine & shuffle
        points_combined = torch.cat((points_label_0_down, points_label_1_down), dim=0)
        labels_combined = torch.cat((labels_label_0_down, labels_label_1_down), dim=0)


        indices_combined = torch.randperm(len(points_combined))
        points_combined = points_combined[indices_combined]
        labels_combined = labels_combined[indices_combined]

        return points_combined, labels_combined


    def prepare_pointcloud_offline(self, pc_query, pc_query_label):

        # subsample rate
        if self.query_sample_rate != 1.0:
            pc_query,pc_query_label = self.sample_pc_query(pc_query,pc_query_label)     
            return pc_query, pc_query_label
        else:
            return pc_query, pc_query_label

                                
    def prepare_train_data(self, input_dict, input_dict_index):

        input_dict = self.T_GridSample(input_dict, input_dict_index['idx_select']) 
        input_dict = self.T_ToTensor(input_dict)
        input_dict_index = self.T_ToTensor(input_dict_index)

        coord = input_dict["coord"] # 已经是grid_size采样后的
        grid_coord = input_dict["grid_coord"]
        color = input_dict["color"]
        pc_query = input_dict['pc_query_0.05']
        pc_query_label = input_dict['pc_query_label_0.05']

        fps_coord = coord[input_dict_index['fps_select']]

        # downsample
        pc_query, pc_query_label = self.prepare_pointcloud_offline(pc_query,pc_query_label)

        result_dict = {
            'coord': coord,
            'grid_coord': grid_coord,
            'color': color,
            'offset': torch.tensor([coord.shape[0]]),
            'feat': coord,
            'pc_query': pc_query,
            'pc_query_label': pc_query_label,
            'pc_query_offset': torch.tensor([pc_query.shape[0]]),
            'fps_coord': fps_coord,
            'fps_index': input_dict_index['fps_select'],
            'data_type': 1,
        }            
        return result_dict
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch % self.epoch_num

    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        input_dict_index = torch.load(self.indexset[self.current_epoch][index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict, input_dict_index)
        return result_dict

    def __len__(self):
        return len(self.samples)
    
# offline fps dataset fps/pcd
@DATASETS.register_module()
class DatasetForScanNetV4(Dataset):
    def __init__(
        self,
        data_root: str,
        fps_root: str,
        data_list_name: str='data_list.txt',
        query_sample_rate: float=1.0,
        max_query_number: int=100000,
        Gridsample_mode: str='train',
        grid_size: float=0.05,
    ):

        """
        Args:
            data_root: Path to root directory containing .pt data
            fps_root: Path to root directory containing 
            data_list_name: File containing list of sample paths (relative or absolute)
            query_sample_rate: Sampling ratio for pc_query
            max_query_number: Max number of points to sample from each label class
            Gridsample_mode: 'train' or 'val'
            grid_size: voxel size for GridSample
        """

        self.data_root = data_root
        self.fps_root = fps_root
        self.samples = []
        self.samples_fps = []
        self.data_list_name = data_list_name
        with open(data_list_name, 'r') as f:
            for p in f.readlines():
                self.samples.append(os.path.join(self.data_root, p.strip()))
                self.samples_fps.append(os.path.join(self.fps_root, p.strip()))
        self.query_sample_rate = query_sample_rate
        self.max_query_number = max_query_number
        # data aug 
        self.T_GridSample = GridSample(
                        grid_size = grid_size,
                        hash_type="fnv",
                        mode = Gridsample_mode,
                        keys=("coord", "color","normal"),
                        return_grid_coord=True
                        )
        self.T_ToTensor = ToTensor()

    def sample_pc_query(self, points, labels):

        points_label_0 = points[labels == 0]
        points_label_1 = points[labels == 1]

        # deal with label 0
        num_points_label_0 = len(points_label_0)
        indices_label_0 = torch.randperm(num_points_label_0)[:min(self.max_query_number,int(num_points_label_0 * self.query_sample_rate))]
        points_label_0_down = points_label_0[indices_label_0]
        labels_label_0_down = torch.zeros(len(indices_label_0), dtype=torch.float32)

        # deal with label 1
        num_points_label_1 = len(points_label_1)
        indices_label_1 = torch.randperm(num_points_label_1)[:min(self.max_query_number,int(num_points_label_1 * self.query_sample_rate))]
        points_label_1_down = points_label_1[indices_label_1]
        labels_label_1_down = torch.ones(len(indices_label_1), dtype=torch.float32)


        # combine & shuffle
        points_combined = torch.cat((points_label_0_down, points_label_1_down), dim=0)
        labels_combined = torch.cat((labels_label_0_down, labels_label_1_down), dim=0)


        indices_combined = torch.randperm(len(points_combined))
        points_combined = points_combined[indices_combined]
        labels_combined = labels_combined[indices_combined]

        return points_combined, labels_combined


    def prepare_pointcloud_offline(self, pc_query, pc_query_label):

        # subsample rate
        if self.query_sample_rate != 1.0:
            pc_query,pc_query_label = self.sample_pc_query(pc_query,pc_query_label)     
            return pc_query, pc_query_label
        else:
            return pc_query, pc_query_label

                                
    def prepare_train_data(self, input_dict, input_fps_dict):

        input_dict = self.T_GridSample(input_dict)
        input_dict = self.T_ToTensor(input_dict)

        coord = input_dict["coord"]
        grid_coord = input_dict["grid_coord"]
        color = input_dict["color"]
        pc_query = input_dict['pc_query_0.05']
        pc_query_label = input_dict['pc_query_label_0.05']
        fps_coord = input_fps_dict['fps_coord']
        fps_grid_coord = input_fps_dict['grid_coord']
        # downsample
        pc_query, pc_query_label = self.prepare_pointcloud_offline(pc_query,pc_query_label)

        result_dict = {
            'coord': coord,
            'grid_coord': grid_coord,
            'color': color,
            'offset': torch.tensor([coord.shape[0]]),
            'feat': coord,
            'pc_query': pc_query,
            'pc_query_label': pc_query_label,
            'pc_query_offset': torch.tensor([pc_query.shape[0]]),
            'fps_coord': fps_coord,
            'fps_grid_coord': fps_grid_coord,
            'fps_feat': fps_coord,
            'data_type': 2,
        }            
        return result_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        input_fps_dict = torch.load(self.samples_fps[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict, input_fps_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)

@DATASETS.register_module()
class DatasetForScanNetV6(Dataset):
    def __init__(self, data_root: str, data_list_name: str, fps_root: str, **kwargs):
        """
        Args:
            data_root: Path to root directory containing .pt data
            data_list_name: File containing list of sample paths (relative or absolute)
            kwargs:

        """

        self.data_root = data_root
        self.fps_root = fps_root
        
        with open(data_list_name, 'r') as f:
            file_list = f.readlines()
            self.samples = [os.path.join(self.data_root, p.strip()) for p in file_list]
            self.samples_fps = [os.path.join(self.fps_root, p.strip()) for p in file_list]
            # print(len(self.samples), len(self.samples_fps))
            # exit(-1)
        self.data_list_name = data_list_name
        
        ## current voxel res
        self.vs_res = kwargs.get("vs_res",128)
        ## set default values and update with kwargs
        ## - downsample decoder query during training
        self.query_sample_rate = kwargs.get("query_sample_rate",1.0)
        self.max_query_number = kwargs.get("max_query_number",100000)

        ## - class-balance-sampling
        self.query_class_balance_sample = kwargs.get("query_class_balance_sample", False)
        self.pos_neg_ratio = kwargs.get("pos_neg_ratio", 1/2)
        self.max_pos_keep = kwargs.get("max_pos_keep", 100000)
        self.data_list_name = data_list_name

        ## - gridsample param
        grid_size = kwargs.get("grid_size", 2.0 / 128)
        grid_mode = kwargs.get("grid_mode", 'train') ## same with ptv3

        # #data aug 
        self.T_GridSample = GridSample(
                        grid_size=grid_size,
                        hash_type="fnv",
                        mode = grid_mode,
                        keys=("coord", "color","normal"),
                        return_grid_coord=True
                        )
        self.T_ToTensor = ToTensor()

    def sample_pc_query(self, points, labels):

        points_label_0 = points[labels == 0]
        points_label_1 = points[labels == 1]

        # deal with label 0
        num_points_label_0 = len(points_label_0)
        if self.max_query_number == 0:
            indices_label_0 = torch.randperm(num_points_label_0)[:(int(num_points_label_0 * self.query_sample_rate))]
        else:
            indices_label_0 = torch.randperm(num_points_label_0)[:min(self.max_query_number,int(num_points_label_0 * self.query_sample_rate))]
        points_label_0_down = points_label_0[indices_label_0]
        labels_label_0_down = torch.zeros(len(indices_label_0), dtype=torch.float32)

        # deal with label 1
        num_points_label_1 = len(points_label_1)
        if self.max_query_number == 0:
            indices_label_1 = torch.randperm(num_points_label_1)[:(int(num_points_label_1 * self.query_sample_rate))]
        else:
            indices_label_1 = torch.randperm(num_points_label_1)[:min(self.max_query_number,int(num_points_label_1 * self.query_sample_rate))]
        points_label_1_down = points_label_1[indices_label_1]
        labels_label_1_down = torch.ones(len(indices_label_1), dtype=torch.float32)


        # combine & shuffle
        points_combined = torch.cat((points_label_0_down, points_label_1_down), dim=0)
        labels_combined = torch.cat((labels_label_0_down, labels_label_1_down), dim=0)


        indices_combined = torch.randperm(len(points_combined))
        points_combined = points_combined[indices_combined]
        labels_combined = labels_combined[indices_combined]

        return points_combined, labels_combined


    def sample_pc_query_classblance(self, points, labels, pos_neg_ratio=1/3, max_pos_keep=1000):
        """
        Subsample positive samples if too many, and undersample negative samples accordingly to maintain ratio.
        
        Args:
            points: [N, 3]
            labels: [N] - binary 0 or 1
            pos_neg_ratio: positive:negative ratio (e.g., 1:3)
            max_pos_keep: maximum number of positive samples to keep
        """
        points_label_0 = points[labels == 0]  # negative samples
        points_label_1 = points[labels == 1]  # positive samples

        # handle positive samples
        num_points_label_1 = len(points_label_1)
        if num_points_label_1 == 0:
            return points_label_0[:1], torch.zeros(1)  # fallback

        if num_points_label_1 > max_pos_keep:
            indices_label_1 = torch.randperm(num_points_label_1)[:max_pos_keep]
            points_label_1_down = points_label_1[indices_label_1]
        else:
            points_label_1_down = points_label_1

        labels_label_1_down = torch.ones(len(points_label_1_down), dtype=torch.float32)

        # compute target negative count
        num_pos_used = len(points_label_1_down)
        target_neg = int(num_pos_used / pos_neg_ratio)

        num_points_label_0 = len(points_label_0)
        if target_neg > num_points_label_0:
            target_neg = num_points_label_0

        indices_label_0 = torch.randperm(num_points_label_0)[:target_neg]
        points_label_0_down = points_label_0[indices_label_0]
        labels_label_0_down = torch.zeros(len(indices_label_0), dtype=torch.float32)

        # combine & shuffle
        points_combined = torch.cat((points_label_0_down, points_label_1_down), dim=0)
        labels_combined = torch.cat((labels_label_0_down, labels_label_1_down), dim=0)

        indices_combined = torch.randperm(len(points_combined))
        points_combined = points_combined[indices_combined]
        labels_combined = labels_combined[indices_combined]

        return points_combined, labels_combined



    def prepare_pointcloud_offline(self, pc_query,pc_query_label):

        # subsample rate
        if self.query_sample_rate == 1.0:
            return pc_query, pc_query_label
        if self.query_class_balance_sample:
            return self.sample_pc_query_classblance(
                pc_query, pc_query_label,
                pos_neg_ratio=self.pos_neg_ratio,
                max_pos_keep=self.max_pos_keep
            )
        return self.sample_pc_query(pc_query, pc_query_label)

                                
    def prepare_train_data(self, input_dict, input_fps_dict):
        
        ## gridsample for input
        input_dict = self.T_GridSample(input_dict)
        input_dict = self.T_ToTensor(input_dict)
        input_fps_dict = self.T_ToTensor(input_fps_dict)

        # input pcd
        coord = input_dict["coord"]
        grid_coord = input_dict["grid_coord"]
        color = input_dict["color"]
        scene_id = input_dict["scene_id"]
        fps_coord = input_fps_dict['fps_coord']
        fps_grid_coord = input_fps_dict['grid_coord']

        result_dict = {
            'coord': coord,
            'grid_coord': grid_coord,
            'color': color,
            'scene_id': scene_id,
            'offset': torch.tensor([coord.shape[0]]),
            'feat': coord,
            'data_type': 2,
            'fps_grid_coord': fps_grid_coord,
            'fps_coord': fps_coord,
            'fps_feat': fps_coord
        }    

        if isinstance(self.vs_res,list):
            for vs in self.vs_res[:-1]:
                result_dict[f'pc_query_{vs}'] = input_dict[f'pc_query_{vs}']
                result_dict[f'pc_query_label_fine_prop_{vs}'] = input_dict[f'pc_query_label_fine_prop_{vs}']
                result_dict[f'pc_query_grid_{vs}'] = torch.stack(torch.meshgrid([torch.arange(vs)] * 3, indexing='ij'), dim=-1).reshape(-1, 3).long()
            result_dict[f'pc_query_{self.vs_res[-1]}'] = input_dict[f'pc_query_{self.vs_res[-1]}']
            result_dict[f'pc_query_label_{self.vs_res[-1]}'] = input_dict[f'pc_query_label_{self.vs_res[-1]}']
            result_dict[f'pc_query_grid_{self.vs_res[-1]}'] = torch.stack(torch.meshgrid([torch.arange(self.vs_res[-1])] * 3, indexing='ij'), dim=-1).reshape(-1, 3).long()
                        
        # # decoder query
        # pc_query = input_dict[f'pc_query_{self.vs_res}']
        # pc_query_label = input_dict[f'pc_query_label_{self.vs_res}']

        # # downsample
        # pc_query, pc_query_label = self.prepare_pointcloud_offline(pc_query,pc_query_label)
          
        return result_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        input_fps_dict = torch.load(self.samples_fps[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict, input_fps_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)