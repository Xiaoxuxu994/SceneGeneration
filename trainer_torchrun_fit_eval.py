import argparse
from collections import defaultdict
from easydict import EasyDict as edict
from gc import collect
from glob import glob
import json
import lpips
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

from src.utils.train_utils import saveRuntimeCode


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


@torch.autocast('npu', torch.bfloat16)
def train_epoch(
    model: torch.nn.parallel.DistributedDataParallel,
    sampler: DistributedSampler,
    loader: DataLoader,
    epoch_range: tuple[int, int],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter
) -> int:
    rank0 = not dist.get_rank()
    log_step = epoch_range[0] * len(loader)
    model.eval()
    model.module.eval()
    model.module.lpips_loss.eval().requires_grad_(False)

    for epoch in range(*epoch_range):
        if sampler is not None: sampler.set_epoch(epoch)
        model.module.current_epoch = epoch
        pbar = tqdm(loader, f'Epoch {epoch}', disable=not rank0)
        for inputs in pbar:
            inputs = {k: v[0].to('npu', torch.bool if v.dtype == torch.bool else torch.bfloat16)
                      if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            loss, log_dict = model.forward(inputs, 'train')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.module.opt.trainer.gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            log_step += 1
            log_values = torch.stack(list(log_dict.values())).detach()
            dist.reduce(log_values, 0, dist.ReduceOp.AVG)
            if rank0:
                for k, v in zip(log_dict.keys(), log_values):
                    log_dict[k] = v = v.item()
                    writer.add_scalar(k, v, log_step, new_style=True)
                pbar.set_postfix(log_dict)
    if rank0: writer.flush()


@torch.inference_mode()
@torch.autocast('npu', torch.bfloat16)
def val_epoch(
    model: torch.nn.parallel.DistributedDataParallel,
    loader: DataLoader,
    log_step: int,
    writer: SummaryWriter,
    mode: str = 'val'
):
    rank0 = not dist.get_rank()
    val_size = torch.tensor(len(loader), device='npu')
    sum_val_dict = defaultdict(float)
    pbar = tqdm(loader, 'Validation', disable=not rank0)
    model.eval()
    model.module.eval()
    for inputs in pbar:
        inputs = {k: v[0].to('npu', torch.bool if v.dtype == torch.bool else torch.bfloat16)
                  if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        _, log_dict = model.forward(inputs, mode)
        for k, v in log_dict.items():
            sum_val_dict[k] += v
            log_dict[k] = v.item()
        if rank0: pbar.set_postfix(log_dict)
    dist.reduce(val_size, 0)
    for k in sum_val_dict: dist.reduce(sum_val_dict[k], 0)
    if rank0:
        sum_val_dict = {k: v.item() / val_size.item() for k, v in sum_val_dict.items()}
        for k, v in sum_val_dict.items(): writer.add_scalar(k, v, log_step, new_style=True)
        writer.flush()


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
        from gsdecoderv1 import GSDecoderModel as ModelClass
    elif 'configs/Refiner/' in args.config_path:
        from train.datasets import RefinerDataset as DataClass
        from refiner import RefinerModel as ModelClass
    elif 'configs/Densifier/' in args.config_path:
        from train.datasets import DensifierDataset as DataClass
        raise NotImplementedError
    else: raise NotImplementedError

    if opt.test:
        test_loaders = []
        metadatas = [val_metadata] # val_metadata,  train_metadata
        stages = ['test_val_nv'] # 'test_val_sv',  'test_train_nv', 
        for idx, (test_metadata, stage) in enumerate(zip(metadatas, stages)):
            test_dataset = DataClass(test_metadata, opt, stage=stage)
            test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size,
                                        shuffle=False, num_workers=4, pin_memory=True,
                                        persistent_workers=True)
            test_loaders.append(test_loader)
            logger.info(f"Test Dataset [{idx}] contains {len(test_loader.dataset)}")
            os.makedirs(os.path.join(opt.output_dir, 'test', str(idx)), exist_ok=True)
    else:
        if opt.shard_data:  # manual data sharding and caching
            rank = dist.get_rank()
            n_samples = len(train_metadata) // dist.get_world_size()
            train_dataset = DataClass(train_metadata[rank * n_samples: (rank + 1) * n_samples],
                                      opt, stage='train', shard_data=True)
            train_sampler = None
            train_loader = DataLoader(train_dataset, opt.trainer.batch_size, True,
                                    num_workers=1, collate_fn=collate_skip_none,
                                    pin_memory=True, persistent_workers=True)
        else:
            train_dataset = DataClass(train_metadata, opt, stage='train')
            train_sampler = DistributedSampler(train_dataset, drop_last=True)
            train_loader = DataLoader(train_dataset, opt.trainer.batch_size,
                                    sampler=train_sampler, num_workers=4,
                                    collate_fn=collate_skip_none,
                                    pin_memory=True, persistent_workers=True)
        val_dataset = DataClass(val_metadata, opt, stage='val')
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, opt.trainer.batch_size,
                                sampler=val_sampler, num_workers=4,
                                collate_fn=collate_skip_none,
                                pin_memory=True, persistent_workers=True)
        
    print(f'[INFO] setting input_info_type: {opt.models.args.enc_infos}')

    # ======================================================
    # 3. Build model
    # ======================================================

    # model = ModelClass(opt, load_pretrained='./ckpts/best-epoch=515-val_avg_psnr=0.00.ckpt').to(f'npu', torch.bfloat16)
    model = ModelClass(opt).to(f'npu', torch.bfloat16)
    optimizer, scheduler = model.configure_optimizers()
    optimizer, scheduler = optimizer[0], scheduler[0]
    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)

    # if opt.test:
    #     prepared_test_loaders = []
    #     for test_loader_idx, test_loader in enumerate(test_loaders):
    #         testloader_prep = accelerator.prepare_data_loader(test_loader, device_placement=False)
    #         prepared_test_loaders.append((test_loader_idx, testloader_prep))
    #     test_loaders = prepared_test_loaders
    
    print('[INFO] voxel_align_ratio: ', opt.models.voxel_align_ratio)

    epoch = 0
    if opt.ckpt:  # finetune
        ckpt = torch.load(opt.ckpt, 'npu')
        print(f'[INFO] load from {opt.ckpt}')
        model.load_state_dict(ckpt[0])
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
    model.lpips_loss = lpips.LPIPS(net='vgg').eval().requires_grad_(False).to('npu', torch.float)
    logger.info(f"[INFO] Training starts at epoch {epoch}")

    # =======================================================
    # 4. Training Loop
    # =======================================================

    if rank0:
        logger.info(f"[INFO] Training for {opt.trainer.max_epochs} epochs with {len(train_loader)} steps per epoch")
        logger.info(f"[INFO] Evaluating every {opt.trainer.ckpt_every} interval with {len(val_loader)} steps per epoch")
        assert opt.trainer.ckpt_every > 0

    model.module.on_train_start()

    for epoch in range(epoch, int(opt.trainer.max_epochs), opt.trainer.ckpt_every):
        train_epoch(model, train_sampler, train_loader, (epoch, epoch + opt.trainer.ckpt_every), optimizer, scheduler, writer if rank0 else None)
        collect()
        torch_npu.npu.empty_cache()

        epoch += opt.trainer.ckpt_every

        if rank0:
            state_dicts = model.state_dict(), optimizer.state_dict(), scheduler.state_dict()
            torch.save(state_dicts, os.path.join(f'{opt.output_dir}/ckpt/epoch_{epoch:04}.ckpt'))
            del state_dicts

        val_epoch(model, val_loader, len(train_loader) * epoch, writer if rank0 else None)
        collect()
        torch_npu.npu.empty_cache()

    if rank0: writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()