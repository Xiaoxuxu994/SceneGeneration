import os
import torch
from torch.utils.data import Dataset
from src.dataset.transform import *

class ScanNetPPV2Dataset(Dataset):
    def __init__(self, data_root: str, data_list_name: str, aug_cfg: dict): 

        self.data_root = data_root
        self.aug_cfg = aug_cfg
        self.input_dict_list = []

        # read data txt & preload data dict
        with open(data_list_name, 'r') as f:
            self.samples = f.readlines()
        
        for path in self.samples:
            self.input_dict_list.append(torch.load(
                os.path.join(self.data_root,path.rstrip('\n')),map_location='cpu'))
        
        # initialize data aug 
        self.transforms = []

        # normalize coord
        if aug_cfg.get('NormalizeCoord', {}).get('enabled', False):
            self.transforms.append(NormalizeCoord())

        # random scale
        if aug_cfg.get('RandomScale', {}).get('enabled', False):
            self.transforms.append(RandomScale(
                scale = aug_cfg['RandomScale'].get('scale',None)
            ))

        # GridSample
        if aug_cfg.get('GridSample', {}).get('enabled', False):
            self.transforms.append(GridSample(
                grid_size = aug_cfg['GridSample'].get('grid_size',0.05),
                hash_type = aug_cfg['GridSample'].get('hash_type', 'fnv'),
                mode = aug_cfg['GridSample'].get('Gridsample_mode','train'),
                keys = aug_cfg['GridSample'].get('keys', ("coord",)),
                return_grid_coord = aug_cfg['GridSample'].get('return_grid_coord', True),
            ))

        # DecodeQuerySample
        if aug_cfg.get('DecodeQuerySample', {}).get('enabled', False):
            self.transforms.append(DecodeQuerySample(
                sample_rate = aug_cfg['DecodeQuerySample'].get('sample_rate', 0.1),
                max_number = aug_cfg['DecodeQuerySample'].get('max_number', 100000),
                mode = aug_cfg['DecodeQuerySample'].get('mode','train'),
                prefix = aug_cfg['DecodeQuerySample'].get('prefix', '0.05'),
            ))

        ##
        if aug_cfg.get('ToTensor', {}).get('enabled', False):
            self.transforms.append(ToTensor())

        if aug_cfg.get('Collect', {}).get('enabled', False):
            self.transforms.append(Collect(
                keys=aug_cfg['Collect'].get('keys'),
                feat_keys=aug_cfg['Collect'].get('feat_keys'),
            ))
                                
    def prepare_train_data(self, input_dict):
        for transform in self.transforms:
            input_dict = transform(input_dict)
        return input_dict
    
    def __getitem__(self, index):
        input_dict = self.input_dict_list[index]
        return self.prepare_train_data(input_dict)

    def __len__(self):
        return len(self.samples)