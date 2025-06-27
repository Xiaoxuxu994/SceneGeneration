import os
import torch
from torch.utils.data import Dataset
from src.dataset.transform import *
from src.registry import DATASETS
from .util import sample_farthest_points
import time

@DATASETS.register_module()
class DatasetForScanNetV2(Dataset):
    def __init__(
        self,
        data_root: str,
        fps_root: str='',
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
        pc_query = input_dict['pc_query_0.05']
        pc_query_label = input_dict['pc_query_label_0.05']

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
            'data_type': 0,
        }            
        return result_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)

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
            index_root: Path to root directory containing .pth index (epoch)
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


class AdaptiveSampler:
    def __init__(self, init_threshold=120000):
        self.threshold = init_threshold
        self.history = []

    def update(self, num_points, time_cost, device):
        self.history.append((num_points, time_cost, device))
        if len(self.history) >= 20:
            self._adjust_threshold()

    def _adjust_threshold(self):
        import numpy as np
        cpu = [(n, t) for n, t, d in self.history if d == 'cpu']
        npu = [(n, t) for n, t, d in self.history if d == 'npu']
        if not cpu or not npu:
            return
        for split in range(80000, 200000, 5000):
            cpu_avg = np.mean([t for n, t in cpu if n < split] or [1e10])
            npu_avg = np.mean([t for n, t in npu if n >= split] or [1e10])
            if npu_avg < cpu_avg:
                self.threshold = split
                break
        self.history.clear()

    def use_npu(self, num_points):
        return num_points >= self.threshold

@DATASETS.register_module()
class DatasetForScanNetV5(Dataset):
    def __init__(
        self,
        data_root: str,
        fps_root: str,
        data_list_name: str='data_list.txt',
        query_sample_rate: float=1.0,
        max_query_number: int=100000,
        Gridsample_mode: str='train',
        grid_size: float=0.05,
        latent_num: int=2048,
        latent_random_start: bool=True,
        policy=None,
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
        self.latent_num = latent_num
        self.latent_random_start = latent_random_start
        self.policy = policy or AdaptiveSampler()

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
        pc_query = input_dict['pc_query_0.05']
        pc_query_label = input_dict['pc_query_label_0.05']

        # downsample query pcd
        pc_query, pc_query_label = self.prepare_pointcloud_offline(pc_query,pc_query_label)

        # downsample latent query
        if not self.policy.use_npu(coord.shape[0]):
            start = time.time()
            idx = sample_farthest_points(coord.unsqueeze(0), lengths=torch.tensor([coord.shape[0]], dtype=torch.long), K=self.latent_num, random_start_point=self.latent_random_start).view(-1)
            duration = time.time() - start
            self.policy.update(coord.shape[0], duration, device='cpu')
            fps_index = idx
        else:
            fps_index = None
        fps_index = None
        
        result_dict = {
            'fps_index': fps_index,
            'coord': coord,
            'grid_coord': grid_coord,
            'color': color,
            'offset': torch.tensor([coord.shape[0]]),
            'feat': coord,
            'pc_query': pc_query,
            'pc_query_label': pc_query_label,
            'pc_query_offset': torch.tensor([pc_query.shape[0]]),
            'data_type': 3,
        }            
        return result_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)
