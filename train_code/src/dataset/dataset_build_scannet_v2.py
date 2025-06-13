import os
import torch
from torch.utils.data import Dataset
from src.dataset.transform import *
 
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
            'pc_query_label': pc_query_label ,
        }            
        return result_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)
