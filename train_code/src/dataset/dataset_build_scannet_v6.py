import os
import torch
from typing import Iterable
from torch.utils.data import Dataset
from src.dataset.transform import *
 
class DatasetForScanNetV6(Dataset):
    def __init__(self, data_root: str, data_list_name: str, **kwargs):
        """
        Args:
            data_root: Path to root directory containing .pt data
            data_list_name: File containing list of sample paths (relative or absolute)
            kwargs:

        """

        self.data_root = data_root
        with open(data_list_name, 'r') as f:
            self.samples = [os.path.join(self.data_root, p.strip()) for p in f.readlines()]
        
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

                                
    def prepare_train_data(self, input_dict):
        
        ## gridsample for input
        input_dict = self.T_GridSample(input_dict)
        input_dict = self.T_ToTensor(input_dict)

        # input pcd
        coord = input_dict["coord"]
        grid_coord = input_dict["grid_coord"]
        color = input_dict["color"]
        scene_id = input_dict["scene_id"]

        result_dict = {
            'coord': coord,
            'grid_coord': grid_coord,
            'color': color,
            'scene_id': scene_id,
            'offset': torch.tensor([coord.shape[0]]),
            'feat': coord,
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
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)