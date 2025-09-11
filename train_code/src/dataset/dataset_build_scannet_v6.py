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


class DatasetForScanNetV6DEBUGAUG(Dataset):
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
        
        self.vs_res = kwargs.get("vs_res",[128])
        aug_cfg = kwargs.get("transform",{})
        self.transforms = []
        # initialize data aug 
        if aug_cfg:
            # random dropout
            if aug_cfg.get('RandomDropout', {}).get('enabled', False):
                self.transforms.append(RandomDropout(
                    dropout_ratio = aug_cfg['RandomDropout'].get('dropout_ratio',0.2),
                    dropout_application_ratio = aug_cfg['RandomDropout'].get('dropout_application_ratio',0.2)
                ))

            # random scale
            if aug_cfg.get('RandomScale', {}).get('enabled', False):
                self.transforms.append(RandomScale(
                    scale = aug_cfg['RandomScale'].get('scale',None),
                    vs_list = aug_cfg['RandomScale'].get('vs_list',[128])
                ))

            # random flip
            if aug_cfg.get('RandomFlip', {}).get('enabled', False):
                self.transforms.append(RandomFlip(
                    p = aug_cfg['RandomFlip'].get('p',None),
                    vs_list = aug_cfg['RandomFlip'].get('vs_list',[128])
                ))

            # GridSample
            if aug_cfg.get('GridSample', {}).get('enabled', False):
                self.transforms.append(GridSample(
                    grid_size = aug_cfg['GridSample'].get('grid_size',0.05),
                    hash_type = aug_cfg['GridSample'].get('hash_type', 'fnv'),
                    mode = aug_cfg['GridSample'].get('mode','train'),
                    keys = aug_cfg['GridSample'].get('keys', ("coord",)),
                    return_grid_coord = aug_cfg['GridSample'].get('return_grid_coord', False),
                    return_min_coord = aug_cfg['GridSample'].get('return_min_coord', False),
                ))

            # UniformDownSample
            if aug_cfg.get('UniformDownSample', {}).get('enabled', False):
                self.transforms.append(UniformDownSample(
                    point_max = aug_cfg['UniformDownSample'].get('point_max',50000),
                    sample_rate = aug_cfg['UniformDownSample'].get('sample_rate', None),
                    random = aug_cfg['UniformDownSample'].get('random',True)
                ))

            # ShufflePoint
            if aug_cfg.get('ShufflePoint', {}).get('enabled', False):
                self.transforms.append(ShufflePoint())

            # Decoderquery sample
            if aug_cfg.get('DecodeQuerySample', {}).get('enabled', False):
                self.transforms.append(DecodeQuerySample(
                    sample_rate = aug_cfg['DecodeQuerySample'].get('sample_rate',0.1),
                    max_number = aug_cfg['DecodeQuerySample'].get('max_number', 200000),
                    vs_list = aug_cfg['DecodeQuerySample'].get('vs_list',[128]),
                ))

            # To tensor
            if aug_cfg.get('ToTensor', {}).get('enabled', False):
                self.transforms.append(ToTensor())

            # collect_fn
            if aug_cfg.get('Collect', {}).get('enabled', False):
                self.transforms.append(Collect(
                    keys=aug_cfg['Collect'].get('keys'),
                    feat_keys=aug_cfg['Collect'].get('feat_keys'),
                ))


        ##  cache input query center and grid
        self.grid_index_dict = {}
        self.grid_center_dict = {}
        vs_list = self.vs_res if isinstance(self.vs_res, list) else [self.vs_res]
        for vs in vs_list:
            grid_idx, center = self._generate_dense_grid_index_and_center(vs)
            self.grid_index_dict[vs] = grid_idx
            self.grid_center_dict[vs] = center

    @staticmethod
    def _generate_dense_grid_index_and_center(resolution):
        """
        Args:
            resolution: int, number of divisions along one axis (e.g. 128)
        Returns:
            grid_idx: (N, 3), torch.LongTensor
            center_coords: (N, 3), torch.FloatTensor
        """
        vs_size = 2.0 / resolution
        num_voxels = resolution + 1  # extend half voxel
        origin = np.array([-1.0 - vs_size / 2] * 3, dtype=np.float32)

        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(num_voxels),
            np.arange(num_voxels),
            np.arange(num_voxels),
            indexing='ij'
        )
        grid_idx = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.int32)
        center_coords = origin + (grid_idx + 0.5) * vs_size

        return grid_idx, center_coords.astype(np.float32)
        
    def prepare_train_data(self, input_dict):
        
        ## get query center and grid idx online
        if isinstance(self.vs_res,list):
            for vs in self.vs_res:
                input_dict[f'pc_query_grid_{vs}'] = self.grid_index_dict[vs]
                input_dict[f'pc_query_{vs}'] = self.grid_center_dict[vs]
        
        for transform in self.transforms:
            input_dict = transform(input_dict)
        return input_dict
    
    def __getitem__(self, index):
        input_dict = torch.load(self.samples[index], map_location='cpu')
        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)
    

class DatasetForScanNetV6DEBUGAUGSDF(Dataset):
    def __init__(self, data_root: str, data_list_name: str, **kwargs):
        """
        Args:
            data_root: Path to root directory containing .pt data
            data_list_name: File containing list of sample paths (relative or absolute)
            kwargs:

        """

        self.data_root = data_root
        self.datatype = kwargs.get("datatype",'pth')
        with open(data_list_name, 'r') as f:
            if self.datatype == 'pth':
                self.samples = [os.path.join(self.data_root, p.strip()) for p in f.readlines()] ## pth
            elif self.datatype == 'npz':
                self.samples = [os.path.join(self.data_root, p.strip().replace('.pth','.npz')) for p in f.readlines()] ## npz
            else:
                raise ValueError(f"[ERROR] Unrecognized datatype: {self.datatype}. Supported types are ['pth', 'npz'].")

        self.vs_res = kwargs.get("vs_res",[128])
        aug_cfg = kwargs.get("transform",{})
        self.transforms = []
        # initialize data aug 
        if aug_cfg:
            # random dropout
            if aug_cfg.get('RandomDropout', {}).get('enabled', False):
                self.transforms.append(RandomDropout(
                    dropout_ratio = aug_cfg['RandomDropout'].get('dropout_ratio',0.2),
                    dropout_application_ratio = aug_cfg['RandomDropout'].get('dropout_application_ratio',0.2)
                ))

            # random scale
            if aug_cfg.get('RandomScale', {}).get('enabled', False):
                self.transforms.append(RandomScale(
                    scale = aug_cfg['RandomScale'].get('scale',None),
                    vs_list = aug_cfg['RandomScale'].get('vs_list',[128])
                ))

            # random flip
            if aug_cfg.get('RandomFlip', {}).get('enabled', False):
                self.transforms.append(RandomFlip(
                    p = aug_cfg['RandomFlip'].get('p',None),
                    vs_list = aug_cfg['RandomFlip'].get('vs_list',[128])
                ))

            # # TruncateSDF
            # if aug_cfg.get('TruncateSDF', {}).get('enabled', False):
            #     self.transforms.append(TruncateSDF(
            #         truncatedist = aug_cfg['TruncateSDF'].get('truncatedist',0.02),
            #         vs_list = aug_cfg['TruncateSDF'].get('vs_list',[128])
            #     ))

            # GridSample
            if aug_cfg.get('GridSample', {}).get('enabled', False):
                self.transforms.append(GridSample(
                    grid_size = aug_cfg['GridSample'].get('grid_size',0.05),
                    hash_type = aug_cfg['GridSample'].get('hash_type', 'fnv'),
                    mode = aug_cfg['GridSample'].get('mode','train'),
                    keys = aug_cfg['GridSample'].get('keys', ("coord",)),
                    return_grid_coord = aug_cfg['GridSample'].get('return_grid_coord', False),
                    return_min_coord = aug_cfg['GridSample'].get('return_min_coord', False),
                ))

            # UniformDownSample
            if aug_cfg.get('UniformDownSample', {}).get('enabled', False):
                self.transforms.append(UniformDownSample(
                    point_max = aug_cfg['UniformDownSample'].get('point_max',50000),
                    sample_rate = aug_cfg['UniformDownSample'].get('sample_rate', None),
                    random = aug_cfg['UniformDownSample'].get('random',True)
                ))

            # ShufflePoint
            if aug_cfg.get('ShufflePoint', {}).get('enabled', False):
                self.transforms.append(ShufflePoint())

            # Decoderquery sample
            if aug_cfg.get('DecodeQuerySampleSDF', {}).get('enabled', False):
                self.transforms.append(DecodeQuerySampleSDF(
                    sample_rate = aug_cfg['DecodeQuerySampleSDF'].get('sample_rate',0.1),
                    max_number = aug_cfg['DecodeQuerySampleSDF'].get('max_number', 200000),
                    vs_list = aug_cfg['DecodeQuerySampleSDF'].get('vs_list',[128]),
                    near_thresh = aug_cfg['DecodeQuerySampleSDF'].get('near_thresh',0.05),
                    near_ratio = aug_cfg['DecodeQuerySampleSDF'].get('near_ratio',0.8),
                ))

            # Decoderquery sample random
            if aug_cfg.get('DecodeQueryRandSampleSDF', {}).get('enabled', False):
                self.transforms.append(DecodeQueryRandSampleSDF(
                    sample_rate = aug_cfg['DecodeQueryRandSampleSDF'].get('sample_rate',0.1),
                    max_number = aug_cfg['DecodeQueryRandSampleSDF'].get('max_number', 200000),
                    vs_list = aug_cfg['DecodeQueryRandSampleSDF'].get('vs_list',[128]),
                ))
                
            # To tensor
            if aug_cfg.get('ToTensor', {}).get('enabled', False):
                self.transforms.append(ToTensor())

            # collect_fn
            if aug_cfg.get('Collect', {}).get('enabled', False):
                self.transforms.append(Collect(
                    keys=aug_cfg['Collect'].get('keys'),
                    offset_keys_dict=aug_cfg['Collect'].get('offset_keys_dict'),
                ))


        ##  cache input query center and grid
        self.grid_index_dict = {}
        self.grid_minbound_dict = {}
        vs_list = self.vs_res if isinstance(self.vs_res, list) else [self.vs_res]
        for vs in vs_list:
            # if vs == 512:
            #     continue
            grid_idx, minbound = self._generate_dense_grid_index_and_minbound(vs)
            self.grid_index_dict[vs] = grid_idx
            self.grid_minbound_dict[vs] = minbound

    @staticmethod
    def _generate_dense_grid_index_and_minbound(resolution):
        """
        Args:
            resolution: int, number of divisions along one axis (e.g. 128)
        Returns:
            grid_idx: (N, 3), int32
            minbound_coords: (N, 3), float32
        """
        vs_size = 2.0 / resolution
        num_voxels = resolution  # extend half voxel

        origin = np.array([-1.0] * 3, dtype=np.float32)

        # 构造 index 网格
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(num_voxels),
            np.arange(num_voxels),
            np.arange(num_voxels),
            indexing='ij'
        )
        grid_idx = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.int32)

        # 计算 voxel 中心坐标
        minbound_coords = origin + (grid_idx) * vs_size

        return grid_idx, minbound_coords.astype(np.float32)

    def index_to_coord(self,grid_idx, resolution):
        grid_idx = grid_idx.reshape(-1, 3)
        vs_size = 2.0 / resolution 
        origin = np.array([-1.0] * 3, dtype=np.float32)
        coords = origin + grid_idx * vs_size
        return coords.astype(np.float32)
            
    def prepare_train_data(self, input_dict):
        
        ## get query center and grid idx online
        if isinstance(self.vs_res,list):
            for vs in self.vs_res:
                if f'pc_query_sdf_{vs}_oct' in input_dict.keys():
                    input_dict[f'pc_query_sdf_{vs}'] = input_dict[f'pc_query_sdf_{vs}_oct']
                    input_dict[f'pc_query_grid_{vs}'] = input_dict[f'pc_query_grid_{vs}_oct']
                    input_dict[f'pc_query_{vs}'] = self.index_to_coord(input_dict[f'pc_query_grid_{vs}'], vs)

                else:
                    input_dict[f'pc_query_grid_{vs}'] = self.grid_index_dict[vs]
                    input_dict[f'pc_query_{vs}'] = self.grid_minbound_dict[vs]
        
        for transform in self.transforms:
            input_dict = transform(input_dict)
        return input_dict
    
    def __getitem__(self, index):
        
        if self.datatype == 'pth':
            # deal with pth
            input_dict = torch.load(self.samples[index], map_location='cpu')
        elif self.datatype == 'npz':
            ## deal with npz
            input_dict = dict(np.load(self.samples[index]))
            input_dict['scene_id'] = input_dict['scene_id'].item()

        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)
    
class DatasetForScanNetV6DEBUGAUGSVF(Dataset):
    def __init__(self, data_root: str, data_list_name: str, **kwargs):
        """
        Args:
            data_root: Path to root directory containing .pt data
            data_list_name: File containing list of sample paths (relative or absolute)
            kwargs:

        """

        self.data_root = data_root
        self.datatype = kwargs.get("datatype",'pth')
        with open(data_list_name, 'r') as f:
            if self.datatype == 'pth':
                self.samples = [os.path.join(self.data_root, p.strip()) for p in f.readlines()] ## pth
            elif self.datatype == 'npz':
                self.samples = [os.path.join(self.data_root, p.strip().replace('.pth','.npz')) for p in f.readlines()] ## npz
            else:
                raise ValueError(f"[ERROR] Unrecognized datatype: {self.datatype}. Supported types are ['pth', 'npz'].")

        self.vs_res = kwargs.get("vs_res",[128])
        aug_cfg = kwargs.get("transform",{})
        self.transforms = []
        # initialize data aug 
        if aug_cfg:
            # random dropout
            if aug_cfg.get('RandomDropout', {}).get('enabled', False):
                self.transforms.append(RandomDropout(
                    dropout_ratio = aug_cfg['RandomDropout'].get('dropout_ratio',0.2),
                    dropout_application_ratio = aug_cfg['RandomDropout'].get('dropout_application_ratio',0.2)
                ))

            # random scale
            if aug_cfg.get('RandomScale', {}).get('enabled', False):
                self.transforms.append(RandomScale(
                    scale = aug_cfg['RandomScale'].get('scale',None),
                    vs_list = aug_cfg['RandomScale'].get('vs_list',[128])
                ))

            # random flip
            if aug_cfg.get('RandomFlip', {}).get('enabled', False):
                self.transforms.append(RandomFlip(
                    p = aug_cfg['RandomFlip'].get('p',None),
                    vs_list = aug_cfg['RandomFlip'].get('vs_list',[128])
                ))

            # # TruncateSDF
            # if aug_cfg.get('TruncateSDF', {}).get('enabled', False):
            #     self.transforms.append(TruncateSDF(
            #         truncatedist = aug_cfg['TruncateSDF'].get('truncatedist',0.02),
            #         vs_list = aug_cfg['TruncateSDF'].get('vs_list',[128])
            #     ))

            # GridSample
            if aug_cfg.get('GridSample', {}).get('enabled', False):
                self.transforms.append(GridSample(
                    grid_size = aug_cfg['GridSample'].get('grid_size',0.05),
                    hash_type = aug_cfg['GridSample'].get('hash_type', 'fnv'),
                    mode = aug_cfg['GridSample'].get('mode','train'),
                    keys = aug_cfg['GridSample'].get('keys', ("coord",)),
                    return_grid_coord = aug_cfg['GridSample'].get('return_grid_coord', False),
                    return_min_coord = aug_cfg['GridSample'].get('return_min_coord', False),
                ))

            # UniformDownSample
            if aug_cfg.get('UniformDownSample', {}).get('enabled', False):
                self.transforms.append(UniformDownSample(
                    point_max = aug_cfg['UniformDownSample'].get('point_max',50000),
                    sample_rate = aug_cfg['UniformDownSample'].get('sample_rate', None),
                    random = aug_cfg['UniformDownSample'].get('random',True)
                ))

            # ShufflePoint
            if aug_cfg.get('ShufflePoint', {}).get('enabled', False):
                self.transforms.append(ShufflePoint())

            # Decoderquery sample
            if aug_cfg.get('DecodeQuerySampleSDF', {}).get('enabled', False):
                self.transforms.append(DecodeQuerySampleSDF(
                    sample_rate = aug_cfg['DecodeQuerySampleSDF'].get('sample_rate',0.1),
                    max_number = aug_cfg['DecodeQuerySampleSDF'].get('max_number', 200000),
                    vs_list = aug_cfg['DecodeQuerySampleSDF'].get('vs_list',[128]),
                    near_thresh = aug_cfg['DecodeQuerySampleSDF'].get('near_thresh',0.05),
                    near_ratio = aug_cfg['DecodeQuerySampleSDF'].get('near_ratio',0.8),
                ))

            # Decoderquery sample random
            if aug_cfg.get('DecodeQueryRandSampleSDF', {}).get('enabled', False):
                self.transforms.append(DecodeQueryRandSampleSDF(
                    sample_rate = aug_cfg['DecodeQueryRandSampleSDF'].get('sample_rate',0.1),
                    max_number = aug_cfg['DecodeQueryRandSampleSDF'].get('max_number', 200000),
                    vs_list = aug_cfg['DecodeQueryRandSampleSDF'].get('vs_list',[128]),
                ))
                
            # To tensor
            if aug_cfg.get('ToTensor', {}).get('enabled', False):
                self.transforms.append(ToTensor())

            # collect_fn
            if aug_cfg.get('Collect', {}).get('enabled', False):
                self.transforms.append(Collect(
                    keys=aug_cfg['Collect'].get('keys'),
                    offset_keys_dict=aug_cfg['Collect'].get('offset_keys_dict'),
                ))


        ##  cache input query center and grid
        self.grid_index_dict = {}
        self.grid_minbound_dict = {}
        vs_list = self.vs_res if isinstance(self.vs_res, list) else [self.vs_res]
        for vs in vs_list:
            # if vs == 512:
            #     continue
            grid_idx, minbound = self._generate_dense_grid_index_and_minbound(vs)
            self.grid_index_dict[vs] = grid_idx
            self.grid_minbound_dict[vs] = minbound

    @staticmethod
    def _generate_dense_grid_index_and_minbound(resolution):
        """
        Args:
            resolution: int, number of divisions along one axis (e.g. 128)
        Returns:
            grid_idx: (N, 3), int32
            minbound_coords: (N, 3), float32
        """
        vs_size = 2.0 / resolution
        num_voxels = resolution  # extend half voxel

        origin = np.array([-1.0] * 3, dtype=np.float32)

        # 构造 index 网格
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(num_voxels),
            np.arange(num_voxels),
            np.arange(num_voxels),
            indexing='ij'
        )
        grid_idx = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.int32)

        # 计算 voxel 中心坐标
        minbound_coords = origin + (grid_idx) * vs_size

        return grid_idx, minbound_coords.astype(np.float32)

    def index_to_coord(self,grid_idx, resolution):
        grid_idx = grid_idx.reshape(-1, 3)
        vs_size = 2.0 / resolution 
        origin = np.array([-1.0] * 3, dtype=np.float32)
        coords = origin + grid_idx * vs_size
        return coords.astype(np.float32)
            
    def prepare_train_data(self, input_dict):
        
        ## get query center and grid idx online
        if isinstance(self.vs_res,list):
            for vs in self.vs_res:
                if f'pc_query_sdf_{vs}_oct' in input_dict.keys():
                    if f'pc_query_grad_{vs}_oct' in input_dict.keys():
                        sdf = input_dict[f'pc_query_sdf_{vs}_oct']   # (N,) or (N,1)
                        grad = input_dict[f'pc_query_grad_{vs}_oct'] # (N,3)

                        if sdf.ndim == 1:   # (N,) → (N,1)
                            sdf = np.expand_dims(sdf, axis=-1)

                        input_dict[f'pc_query_sdf_{vs}'] = np.concatenate([sdf, grad], axis=-1)
                        
                        if f'pc_query_sign_{vs}_oct' in input_dict.keys():
                            sign = input_dict[f'pc_query_sign_{vs}_oct']   # (N,) or (N,1)
                            # transfer to [0,1]
                            tgt_sign = (sign == 1).astype(np.float32)
                            if tgt_sign.ndim == 1:   # (N,) → (N,1)
                                tgt_sign = np.expand_dims(tgt_sign, axis=-1)
                        input_dict[f'pc_query_sdf_{vs}'] = np.concatenate([sdf, grad,tgt_sign], axis=-1)

                    else:
                        input_dict[f'pc_query_sdf_{vs}'] = input_dict[f'pc_query_sdf_{vs}_oct']
                        
                    input_dict[f'pc_query_grid_{vs}'] = input_dict[f'pc_query_grid_{vs}_oct']
                    input_dict[f'pc_query_{vs}'] = self.index_to_coord(input_dict[f'pc_query_grid_{vs}'], vs)

                else:
                    input_dict[f'pc_query_grid_{vs}'] = self.grid_index_dict[vs]
                    input_dict[f'pc_query_{vs}'] = self.grid_minbound_dict[vs]
        
        for transform in self.transforms:
            input_dict = transform(input_dict)
        return input_dict
    
    def __getitem__(self, index):
        
        if self.datatype == 'pth':
            # deal with pth
            input_dict = torch.load(self.samples[index], map_location='cpu')
        elif self.datatype == 'npz':
            ## deal with npz
            input_dict = dict(np.load(self.samples[index]))
            input_dict['scene_id'] = input_dict['scene_id'].item()

        result_dict = self.prepare_train_data(input_dict)
        return result_dict

    def __len__(self):
        return len(self.samples)