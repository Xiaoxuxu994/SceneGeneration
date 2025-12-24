import argparse
from collections import defaultdict
from easydict import EasyDict as edict
from gc import collect
from glob import glob
import json
from loguru import logger
import os
import pandas as pd
import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torch_npu
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, default_collate
from torch.utils.tensorboard import SummaryWriter
import lpips

from src.utils.train_utils import saveRuntimeCode


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


@torch.inference_mode()
@torch.autocast('npu', torch.float)
def test_epoch(
    model,
    loader_lst: list[DataLoader],
    writer: SummaryWriter,
    mode: str = 'val'
):
    rank0 = not dist.get_rank()
    for loader_idx, loader in enumerate(loader_lst):
        pbar = tqdm(loader, 'Test', disable=not rank0)
        model.eval()
        for idx, inputs in enumerate(pbar):
            inputs = {k: v[0].to('npu', torch.bool if v.dtype == torch.bool else torch.float32)
                    if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}            
            # inputs = {k: v[0].to('npu', torch.bool if v.dtype == torch.bool else torch.bfloat16)
            #         if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            model.test_step(inputs, idx, loader_idx)


@record
def main():
    # ======================================================
    # 1. Args & cfg
    # ======================================================
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
    # torch.accelerator.set_device_index(local_rank)
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
    os.makedirs(os.path.join(opt.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'tensorboard'), exist_ok=True)

    rank0 = not dist.get_rank()
    if rank0:
        writer = SummaryWriter(os.path.join(opt.output_dir, 'tensorboard'))
        print("====== cfg: ", opt)
        print("====== exp_dir: ", opt.output_dir)
        logger.remove(handler_id=None)
        logger.add(os.path.join(opt.output_dir, "loss.log"), level="INFO")
        logger.info(f"opt: {opt}")
        logger.info('\n\nConfig:')
        logger.info('=' * 80)
        json_str = json.dumps(opt.__dict__, indent=4)  # 将对象转为格式化的 JSON 字符串
        logger.info(json_str)

        with open(opt.output_dir+'/opt_config.json', 'w') as f: f.write(json_str)
        try: saveRuntimeCode(os.path.join(opt.output_dir, 'backup'))
        except: logger.info(f'save code failed~')

    train_metadata = pd.read_csv(opt.train_metadata)
    val_metadata = pd.read_csv(opt.val_metadata)

    logger.info('\n\n'+'=='*60)
    logger.info(f"[INFO] metadata num: 904, filtered metadata num: {len(train_metadata) + len(val_metadata)}, train_metadata num: {len(train_metadata)}, val_metadata num: {len(val_metadata)}")
    logger.info('=='*60 + '\n\n')

    # ======================================================
    # 2. Build dataset and dataloader
    # ======================================================

    if 'configs/GSDecoder/' in args.config_path:
        from train.datasets import GSDecoderDataset as DataClass
        # from gsdecoder import GSDecoderModel as ModelClass
        from gsdecoderv1_npu import GSDecoderModel as ModelClass
    elif 'configs/Refiner/' in args.config_path:
        from train.datasets import RefinerDataset as DataClass
        from refiner import RefinerModel as ModelClass
    elif 'configs/Densifier/' in args.config_path:
        from train.datasets import DensifierDataset as DataClass
        raise NotImplementedError
    else: raise NotImplementedError

    test_loaders = []
    metadatas = [val_metadata] # val_metadata,  train_metadata
    stages = ['test'] # 'test_val_sv',  'test_train_nv', 
    for idx, (test_metadata, stage) in enumerate(zip(metadatas, stages)):
        test_dataset = DataClass(test_metadata, opt, stage=stage)
        test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True,
                                    persistent_workers=True)
        test_loaders.append(test_loader)
        logger.info(f"Test Dataset [{idx}] contains {len(test_loader.dataset)}")
        os.makedirs(os.path.join(opt.output_dir, 'test', str(idx)), exist_ok=True)

    # ======================================================
    # 3. Build model
    # ======================================================

    # model = ModelClass(opt, load_pretrained='./ckpts/best-epoch=515-val_avg_psnr=0.00.ckpt').to(f'npu', torch.bfloat16)
    model = ModelClass(opt).to(f'npu') # torch.bfloat16)
    optimizer, scheduler = model.configure_optimizers() 
    optimizer, scheduler = optimizer[0], scheduler[0]

    # if opt.test:
    #     prepared_test_loaders = []
    #     for test_loader_idx, test_loader in enumerate(test_loaders):
    #         testloader_prep = accelerator.prepare_data_loader(test_loader, device_placement=False)
    #         prepared_test_loaders.append((test_loader_idx, testloader_prep))
    #     test_loaders = prepared_test_loaders

    epoch = 0
    if opt.ckpt:  # finetune
        ckpt = torch.load(opt.ckpt, 'npu')
        print("===============================================")
        print(opt.ckpt)
        print(ckpt[0].keys())
        print("===============================================")
        #  单卡加载ddpckpt
        state_dict = ckpt[0]['state_dict'] if 'state_dict' in ckpt[0] else ckpt[0]
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            if 'configs/Refiner/' in args.config_path:  # refiner 需要
                new_key = new_key.replace('update_mlp_dg', 'update_module.mlp_dg')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:  # resume
        ckpts = sorted(glob(f'{opt.output_dir}/ckpt/epoch_????.ckpt'))
        if ckpts:
            try:
                ckpt = torch.load(ckpts[-1], 'npu')
                epoch = int(ckpts[-1][-9:-5])
            except:
                try:
                    ckpt = torch.load(ckpts[-2], 'npu')
                    epoch = int(ckpts[-2][-9:-5])
                except: ckpt = None
            if ckpt is not None:
                model.load_state_dict(ckpt[0])
                optimizer.load_state_dict(ckpt[1])
                scheduler.load_state_dict(ckpt[2])
    #if 'configs/Refiner/' in args.config_path:  # refiner 需要
    #    model.eval().requires_grad_(False).to('npu', torch.float)
    # model.eval().to('npu:0')
    # model.lpips_loss = lpips.LPIPS(True, 'vgg').eval().requires_grad_(False).to('npu')
    logger.info(f"[INFO] Training starts at epoch {epoch}")

    # =======================================================
    # 4. Training Loopx
    # =======================================================


    if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.on_test_start()
    else: model.on_test_start()

    test_epoch(model, test_loaders, writer if rank0 else None)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.on_test_epoch_end()
    else: model.on_test_epoch_end()

    torch_npu.npu.empty_cache()

    if rank0: writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()