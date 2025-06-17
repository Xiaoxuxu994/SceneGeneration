import os
import torch
from torch.utils.data import Dataset
from src.dataset.transform import GridSample,ToTensor,NormalizeCoord,ShufflePoint

class DatasetForScanNetV2Test(Dataset):
    def __init__(
        self,
        data_root: str,
        data_list_name: str='data_list.txt',
        query_sample_rate: float=1.0,
        max_query_number: int=100000,
        normalize_coord: bool=False,
        shufflepoint: bool=False,
        Gridsample_mode: str='train',
        grid_size: float=0.05,
    ):
            
        self.data_root = data_root
        self.input_dict_list = []
        with open(data_list_name, 'r') as f:
            self.samples = f.readlines()
        
        for path in self.samples:
            self.input_dict_list.append(torch.load(os.path.join(self.data_root,path.rstrip('\n')),map_location='cpu'))

        self.query_sample_rate = query_sample_rate
        self.max_query_number = max_query_number
        
        # data aug 
        self.normalize_coord = normalize_coord
        self.shufflepoint = shufflepoint
        self.Gridsample_mode = Gridsample_mode
        self.grid_size = grid_size
        # self.T_Centershift_before = CenterShift(apply_z = True)
        if self.normalize_coord:
            self.Normalize_coord = NormalizeCoord()
        self.T_GridSample = GridSample(
                        grid_size=self.grid_size,
                        hash_type="fnv",
                        mode = self.Gridsample_mode,
                        keys=("coord", "color"),
                        return_grid_coord=True,
                        )
        if self.shufflepoint:
            self.shufflepoint = ShufflePoint()
        self.T_ToTensor = ToTensor()

    def sample_pc_query(self, points, labels,sample_rate):
        # 这个函数有三次random打乱。
        points_label_0 = points[labels == 0]
        points_label_1 = points[labels == 1]

        # deal with label 0
        num_points_label_0 = len(points_label_0)
        indices_label_0 = torch.randperm(num_points_label_0)[:int(num_points_label_0 * sample_rate)]
        points_label_0_down = points_label_0[indices_label_0]
        labels_label_0_down = torch.zeros(len(indices_label_0), dtype=torch.float32)

        # deal with label 1
        num_points_label_1 = len(points_label_1)
        indices_label_1 = torch.randperm(num_points_label_1)[:int(num_points_label_1 * sample_rate)]
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
        elif pc_query.shape[0] >= self.max_query_number:
            print(f'Decoder Query {pc_query.shape[0]} is too large, doing downsample 0.1......')
            pc_query,pc_query_label = self.sample_pc_query(pc_query,pc_query_label,sample_rate=0.1)  
            return pc_query, pc_query_label
        else:
            return pc_query, pc_query_label

    def prepare_pointcloud_offline2(self, pc_query,pc_query_label,pc_query_segment):

        # subsample rate
        if self.query_sample_rate != 1.0:
            pc_query,pc_query_label = self.sample_pc_query(pc_query,pc_query_label,pc_query_segment)     
            return pc_query, pc_query_label
        elif pc_query.shape[0] >= self.max_query_number:
            print(f'Decoder Query {pc_query.shape[0]} is too large, doing downsample 0.1......')
            pc_query,pc_query_label = self.sample_pc_query(pc_query,pc_query_label,pc_query_segment,sample_rate=0.1)  
            return pc_query, pc_query_label
        else:
            return pc_query, pc_query_label
                        
    def prepare_train_data(self, input_dict):
        # input_dict = self.T_Centershift_before(input_dict)
        if self.normalize_coord:
            input_dict = self.Normalize_coord(input_dict)

        input_dict = self.T_GridSample(input_dict)
        
        if self.shufflepoint:
            input_dict = self.shufflepoint(input_dict)

        input_dict = self.T_ToTensor(input_dict)

        coord = input_dict["coord"]
        grid_coord = input_dict["grid_coord"]
        color = input_dict["color"]
        scene_id = input_dict["scene_id"]
        offset = torch.tensor([coord.shape[0]])
        # pc_query = input_dict['pc_query']
        # pc_query_label = input_dict['pc_query_label']

        ## 0.05
        pc_query = input_dict['pc_query_0.05']
        pc_query_label = input_dict['pc_query_label_0.05']  

        feat = coord
        
        if 'pc_query_segment' in input_dict.keys():
            pc_query_segment = input_dict['pc_query_segment']
        
        # process query sampling
        if 'pc_query_segment' in input_dict.keys():
            pc_query, pc_query_label,pc_query_segment = self.prepare_pointcloud_offline2(pc_query,pc_query_label,pc_query_segment)
        else:
            pc_query, pc_query_label = self.prepare_pointcloud_offline(pc_query,pc_query_label)

        if 'pc_query_segment' in input_dict.keys():
            result_dict = {
                'coord': coord,
                'grid_coord': grid_coord,
                'color': color,
                'scene_id': scene_id,
                'offset': offset,
                'feat': feat,
                'pc_query': pc_query,
                'pc_query_label': pc_query_label ,
                'pc_query_segment':pc_query_segment,
            }
        else:
            result_dict = {
                'coord': coord,
                'grid_coord': grid_coord,
                'color': color,
                'scene_id': scene_id,
                'offset': offset,
                'feat': feat,
                'pc_query': pc_query,
                'pc_query_label': pc_query_label 
            }

        return result_dict
    

    def __getitem__(self, index):
        input_dict = self.input_dict_list[index]
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)