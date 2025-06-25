import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
from src.registry import DATASETS, MODELS, build_module
import argparse
from src.dataset.transform import *
import os

def prepare_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    **kwargs,
):
    _kwargs = kwargs.copy()
    sampler = RandomSampler(dataset)
    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    def collate_fn(batch):
        merge = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                merge[key] = torch.cat([data[key] for data in batch], dim = 0).unsqueeze(0)
            else:
                merge[key] = batch[0][key]   
        if 'offset' in merge.keys():
            merge['offset'] = torch.cumsum(merge['offset'], dim=1).long()
        if 'pc_query_offset' in merge.keys():
            merge['pc_query_offset'] = torch.cumsum(merge['pc_query_offset'], dim=1).long()         
        return merge
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **_kwargs,
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--id", default='0', type=int)
    args = parser.parse_args()
    id = args.id

    if id == 1: # 测试数据读取
        dataset_cfg = {
            'type': "DatasetForScanNetV2",
            'data_root': '../../data/scannetpp_pcd/train',
            'fps_root':'../../data/scannetpp_pcd/train_fps_fix',
            'data_list_name':'datacache/scannetpp_trainval/nvs_sem_train.txt',
            'query_sample_rate': 0.1,
            'max_query_number': 100000,
            'Gridsample_mode':'train',
            'grid_size':0.05,
        }
        dataset = build_module(dataset_cfg, DATASETS)

        batch_size = 3
        num_workers = 1

        # build dataloader
        dataloader = prepare_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        for epoch in range(0, 1):
            traindataloader_iter = iter(dataloader)
            for i in range(0, 1):
                input_dict = next(traindataloader_iter)
                for key in input_dict.keys():
                    if type(input_dict[key]) is torch.Tensor:
                        print(key, input_dict[key].shape)
                    elif type(input_dict[key]) is list:
                        for obj in input_dict[key]:
                            print(key, type(obj))
                    if key in ['offset','pc_query_offset']:
                        print(input_dict[key])

    if id == 2: #测试多bs的pointtransformer3_v2
        tt = ToTensor()
        grid_size = 0.01
        encoder_dict = dict(
            type = "PointTransformerV3_2",
            in_channels = 3, # ['coord]
            stride=(),
            enc_depths=(1,),
            enc_channels=(512,),
            enc_num_head=(32,),
            enc_patch_size=(32,),
            enable_cpe_block=False,
            enable_flash=False,
            attn_dtype='bf16',
            drop_path=0.3,
            shuffle_orders=False,
            cls_mode=True, #True for enc only; False for enc-dec backbone
        )
        ptv3 = build_module(encoder_dict, MODELS)    
        ptv3.to('cpu')
        with torch.no_grad():
            coord1 = np.random.randn(100,3).astype(np.float32)
            coord2 = np.random.randn(140,3).astype(np.float32)
            coord3 = np.random.randn(80,3).astype(np.float32)
            scaled_coord = coord1 / np.array(grid_size)
            grid_coord1 = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord1.min(0)
            grid_coord1 -= min_coord
            scaled_coord = coord2 / np.array(grid_size)
            grid_coord2 = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord2.min(0)
            grid_coord2 -= min_coord
            scaled_coord = coord3 / np.array(grid_size)
            grid_coord3 = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord3.min(0)
            grid_coord3 -= min_coord


            feat = color = coord = np.concatenate([coord1, coord2, coord3], axis = 0)
            offset = torch.tensor([100,240,320])
            grid_coord = np.concatenate([grid_coord1, grid_coord2, grid_coord3], axis = 0)
            input = dict(
                coord = coord,
                grid_coord = grid_coord,
                feat = feat,
                color = color,
                offset = offset,
            )
            input = tt(input)

            out = ptv3(input)
            print(out.feat.shape)
        exit()
        
