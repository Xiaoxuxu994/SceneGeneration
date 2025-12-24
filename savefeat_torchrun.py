import argparse
from easydict import EasyDict as edict
import json
import os
import pandas as pd
import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torch_npu
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, default_collate


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/Slat_vae_full/exp_04_27_pl.json',
                        help='Path to the config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the output')
    parser.add_argument('--select_scene', type=str, default=None, # 'e0d1ed2159'
                        help='the scenes to calculate')     
    parser.add_argument('--use_depth_mask', action='store_true',
                        help='whether to use depth mask')    
    parser.add_argument('--waitn', type=int, default=9,
                        help='2GPUs=18, 3GPUs=12, 4GPUs=9')
    parser.add_argument('--test', action='store_true', help='whether to test only')
    parser.add_argument('--demo', action='store_true', help='whether to generate demo video only')
    parser.add_argument('--opt', action='append', nargs=2, help='configs to override')
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group('hccl')
    torch_npu.npu.set_device(local_rank)

    with open(args.config_path) as f: config = json.load(f)
    opt = edict(config)
    override = args.opt
    del args.opt
    opt.update(args.__dict__)
    if override is not None:
        for k, v in override:
            try: eval(v)  # number, or list or tuple of numbers
            except: v = '"' + v + '"'  # string
            exec('opt.' + k + '=' + v)

    if opt.select_scene is not None and opt.output_dir is None:
        opt.output_dir = './outputs/single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
    elif opt.select_scene is None and opt.output_dir is None:
        opt.output_dir = './outputs/full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}' # outputs_Slat_vae_full
    else:
        opt.output_dir = opt.output_dir + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
    os.makedirs(os.path.join(opt.output_dir, 'val'), exist_ok=True)

    rank0 = not dist.get_rank()
    metadata = pd.read_csv(opt.train_metadata)

    if 'configs/GSDecoder/' in args.config_path:
        from train.datasets import GSDecoderDataset as DataClass
        from gsdecoderv1 import GSDecoderModel as ModelClass
    elif 'configs/Refiner/' in args.config_path:
        from train.datasets import RefinerDataset as DataClass
        from refiner import RefinerModel as ModelClass
    elif 'configs/Densifier/' in args.config_path:
        from train.datasets import DensifierDataset as DataClass
        raise NotImplementedError
    else: raise NotImplementedError

    val_dataset = DataClass(metadata, opt, stage='val')
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, opt.trainer.batch_size,
                            sampler=val_sampler, num_workers=4,
                            collate_fn=collate_skip_none,
                            pin_memory=True, persistent_workers=True)

    model = ModelClass(opt).to(f'npu', torch.bfloat16)
    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)

    ckpt = torch.load(opt.ckpt, 'npu')
    model.load_state_dict(ckpt[0], False)

    model.module.on_train_start()

    model.zero_grad()
    model.eval().requires_grad_(False)
    with torch.inference_mode(), torch.autocast('npu', torch.bfloat16):
        for inputs in tqdm(val_loader, 'Validation', disable=not rank0):
            inputs = {k: v[0].to('npu', torch.bfloat16 if v.dtype==torch.float else v.dtype)
                      if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model.forward(inputs, 'val')


if __name__ == '__main__':
    main()