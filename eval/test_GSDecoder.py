import accelerate
import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import deepspeed as ds
from easydict import EasyDict as edict
from glob import glob
import json
from loguru import logger
import os
import pandas as pd
from sys import stderr, stdout
import torch
import torch_npu
from tqdm import tqdm, trange
from typing import TextIO
from torch.utils.data import Dataset, DataLoader

from src.utils.train_utils import _filter_dataset, _filter_dataset_new, saveRuntimeCode
from train.datasets import prepare_dataloader

from torch.utils.data._utils.collate import default_collate
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # 可以在 training_step 做判断
    return default_collate(batch)


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
    args = parser.parse_args()

    with open(args.config_path, 'r') as f: config = json.load(f)
    opt = edict(args.__dict__)
    opt.update(config)

    if opt.select_scene is not None and opt.output_dir is None:
        opt.output_dir = './outputs/GSDecoder_single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
    elif opt.select_scene is None and opt.output_dir is None:
        opt.output_dir = './outputs/GSDecoder_full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}' # outputs_Slat_vae_full
    else:
        opt.output_dir = opt.output_dir + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0]+'_accumulateGrad' # outputs_Slat_vae_full
    os.makedirs(os.path.join(opt.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'ckpt'), exist_ok=True)

    accelerator = accelerate.Accelerator(
        log_with='tensorboard',
        project_dir=opt.output_dir,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=True),
            accelerate.InitProcessGroupKwargs('hccl')
        ]
    )

    rank0 = accelerator.is_main_process
    if rank0:  
        accelerator.init_trackers('tensorboard')
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

    new_w, new_h = (1428, 798) if opt.img_resolution is None else opt.img_resolution
    voxel_size = opt.models.args.representation_config.voxel_size

    # 加载 metadata
    if os.path.exists(os.path.join(opt.dataset_dir, f'metadata_norm128_v14.csv')):
        metadata = pd.read_csv(os.path.join(opt.dataset_dir, f'metadata_norm128_v14.csv'))
    else: raise ValueError('metadata.csv not found')
    if opt.select_scene is not None:
        metadata = metadata[metadata['sha256'] == opt.select_scene]
        print(f"[INFO] selected scene is {opt.select_scene}, num_metadata: {len(metadata)}")

    less_coverage_csv = pd.read_csv(os.path.join(opt.dataset_dir, opt.feats_folder_name, opt.feat_model_name, f'less_coverage_scenes_all.csv'))
    # 根据 unconverage_ratio 进行筛选
    threshold = 0.87
    removed_sha256 = less_coverage_csv[less_coverage_csv['unconverage_ratio'] > threshold]['sha256'].tolist()

    excluded_sha256 = ["1204e08f17", "30966f4c6e", "cc5237fd77"] # 验证，这两个场景很差
    excluded_sha256 += ["edbf65c385", "684d4da1fe", "909a9ea5fc"] # OOM voxel太多
    excluded_sha256 += ["45d2e33be1", "9471b8d485", "116456116b", "40b56bf310","2748de13fb"]
    excluded_sha256 += ['2b5ef64cad', '0e75f3c4d9', '4c5c60fa76', '4d451d9c36', '5ea3e738c3', '5fb5d2dbf2', '7cd2ac43b4', '13c3e046d7', '036bce3393', '41b00feddb', '82ff39b7ef', '99fa5c25e1', '623c973c56', '724c40236c', '00777c41d4', '816e996553', '2489b7f4fe', '4117a4f02c', '020312de8d', '13285009a4', 'a4e227f506', 'a23f391ba9', 'a66af39cef', 'aad55d46ad', 'b37c033757', 'b24697b3a1', 'c4c04e6d6c', 'c5f701a8c7', 'c545851c4f', 'c601466b77', 'd415cc449b', 'd918af9c5f', 'e3b3b0d0c7', 'e3c39f4aea', 'e4007ff6b5', 'f248c2bcdc'] # 工厂
    excluded_sha256 += ['0f3474b837', '4f144c7489', '5d152fab1b', '7efa3f6b1b', '9f21bdec45','108ec0b806', '893fb90e89', '3583c4c70f', '20871b98f3', '635852d56e', '4422722c49', '5654092cc2', 'a6e1a3bdda', 'abb6639ceb', 'abf29d2474', 'de3c77cecd', 'f5401524e5', 'fe1733741f'] # storage room  'ada5304e41' 'c07c8a276d' 'd3b7d054c4
    excluded_sha256 += ['1b378f1114', '8e22c48c20', '19cfd590f4', '20ff72df6e', '40b56bf310', '45ba9ea301', '307e3262f1', '1204e08f17', '546292a9db', 'ab046f8faf', 'c4d4cb61f6', 'e3ecd49e2b', 'e909f8213d', 'eeeb9836b8', 'f2e6c43543'] # 不够32
    excluded_sha256 += ['5dd2b849be', 'be0ed6b33c', '546292a9db', '87f6d7d564', 'ab046f8faf']
    train_metadata = metadata[:856]
    train_metadata = train_metadata[
        train_metadata["sha256"].apply(
            lambda sha256: os.path.exists(
                os.path.join(opt.dataset_dir, opt.feats_folder_name, opt.feat_model_name, f'{sha256}.pt')
            )
        )
    ]
    train_metadata = train_metadata[~train_metadata["sha256"].isin(removed_sha256)]
    train_metadata = train_metadata[~train_metadata["sha256"].isin(excluded_sha256)]
    train_metadata = train_metadata[train_metadata['num_voxels_norm128_v14_fix32'] <= 80_000]
    # train_metadata = _filter_dataset(train_metadata, num_voxels_threshold=60_000) # 70_000  metadata[:856]  metadata[:230]
    train_metadata = _filter_dataset_new(train_metadata) # 70_000  metadata[:856]  metadata[:230]
    val_metadata = metadata[856:]
    val_excluded_sha256 = ['f3685d06a9'] # 这个场景之前的anchor点数量超过10k。
    val_metadata = val_metadata[~val_metadata["sha256"].isin(removed_sha256)]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(excluded_sha256)]
    val_metadata = _filter_dataset(metadata[856:], num_voxels_threshold=60_000)   # 70_000  metadata[856:]  metadata[230:]
    val_metadata = val_metadata[~val_metadata["sha256"].isin(val_excluded_sha256)]
                  
    logger.info('\n\n'+'=='*60)
    logger.info(f"[INFO] metadata num: {len(metadata)}, filtered metadata num: {len(train_metadata) + len(val_metadata)}, train_metadata num: {len(train_metadata)}, val_metadata num: {len(val_metadata)}")
    logger.info('=='*60 + '\n\n')

    # ======================================================
    # 2. Build dataset and dataloader
    # ======================================================

    if 'configs/GSDecoder/' in args.config_path:
        from train.datasets import GSDecoderDataset as DataClass
        from gsdecoder import GSDecoderModel as ModelClass
    elif 'configs/Refiner/' in args.config_path:
        from train.datasets import RefinerDataset as DataClass
        from refiner import RefinerModel as ModelClass
    elif 'configs/Densifier/' in args.config_path:
        from train.datasets import DensifierDataset as DataClass
        raise NotImplementedError
    else: raise NotImplementedError

    if opt.select_scene is None:
        train_dataset = DataClass(train_metadata, opt, stage='train')
        val_dataset = DataClass(val_metadata, opt, stage='val')
        # train_loader = prepare_dataloader(train_dataset, 
        #                                   batch_size=opt.trainer.batch_size, 
        #                                   num_workers=8,
        #                                   shuffle=True,
        #                                   drop_last=True,
        #                                   pin_memory=True)
        # val_loader = prepare_dataloader(val_dataset, 
        #                                 batch_size=opt.trainer.batch_size, 
        #                                 num_workers=8,
        #                                 shuffle=False,
        #                                 drop_last=False,
        #                                 pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=False, num_workers=6, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True, 
                                persistent_workers=True )
        if opt.test:
            test_datasets = []
            # train scenes with novel views
            test_metadata_0 = train_metadata
            test_dataset_0 = DataClass(test_metadata_0, opt, stage='test_train_novel')
            test_datasets.append(test_dataset_0)
            # val scenes with novel views
            test_metadata_1 = val_metadata
            test_dataset_1 = DataClass(test_metadata_1, opt, stage='test_val_novel')
            test_datasets.append(test_dataset_1)
            
            test_loaders = []
            for idx, test_dataset in enumerate(test_datasets):
                
                # test_loader = prepare_dataloader(test_dataset, 
                #                             batch_size=opt.trainer.batch_size, 
                #                             num_workers=8,
                #                             shuffle=False,
                #                             drop_last=False,
                #                             pin_memory=True,
                #                             world_size=accelerator.num_processes,
                #                             rank=accelerator.process_index)
                test_loader = DataLoader(test_dataset, batch_size=opt.trainer.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=True, 
                                        persistent_workers=True )
                test_loaders.append(test_loader)

            for idx, test_loader in enumerate(test_loaders):
                logger.info(f"Test Dataset [{idx}] contains {len(test_loader.dataset)}")

    else:
        train_dataset = DataClass(metadata, opt, stage='train')
        val_dataset = DataClass(metadata, opt, stage='val')
        # train_loader = prepare_dataloader(train_dataset, 
        #                                   batch_size=opt.trainer.batch_size, 
        #                                   num_workers=4,
        #                                   shuffle=True,
        #                                   drop_last=True,
        #                                   pin_memory=True)
        # val_loader = prepare_dataloader(val_dataset, 
        #                                 batch_size=opt.trainer.batch_size, 
        #                                 num_workers=4,
        #                                 shuffle=False,
        #                                 drop_last=False,
        #                                 pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=False, num_workers=4, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=False, num_workers=4, pin_memory=True, 
                                  persistent_workers=True,  collate_fn=collate_skip_none)        

    # ======================================================
    # 3. Build model
    # ======================================================

    model = ModelClass(opt)
    optimizer, scheduler = model.configure_optimizers()
    optimizer, scheduler = optimizer[0], scheduler[0]

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    patch_deepspeed(accelerator, model)

    if opt.test:
        prepared_test_loaders = []
        for test_loader_idx, test_loader in enumerate(test_loaders):
            testloader_prep = accelerator.prepare_data_loader(test_loader, device_placement=False)
            prepared_test_loaders.append((test_loader_idx, testloader_prep))
        test_loaders = prepared_test_loaders


    # 3.1. resume training
    # ckpt_list = glob.glob(os.path.join(exp_dir, "epoch*.ckpt"))    
    # if not opt.test and len(ckpt_list) > 0:
    #     resume_from_checkpoint = os.path.join(exp_dir, "last.ckpt")
    # elif opt.ckpt and os.path.exists(opt.ckpt):
    #     last_epoch, last_step, sampler_start_idx = deepspeed_load(model, opt.ckpt)
    #     if rank0: logger.info(f"[INFO] Loaded checkpoint {opt.ckpt} at epoch {last_epoch} step {last_step}")
    #     last_global_step = last_epoch * num_steps_per_epoch + last_step
    #     scheduler.load_state_dict(torch.load(os.path.join(opt.ckpt, f'global_step{last_global_step + 1}/lr_scheduler.pt'))) 
    #     start_step = last_step + 1 if last_step < num_steps_per_epoch - 1 else 0
    #     start_epoch = last_epoch if last_step < num_steps_per_epoch - 1 else last_epoch + 1
        # start_epoch, start_step, sampler_start_idx = load(None, model, optimizer, lr_scheduler, cfg.resume) 

    # =======================================================
    # 4. Training Loop
    # =======================================================

    num_steps_per_epoch = len(train_loader)
    num_steps_per_epoch_val = len(val_loader)
    log_step = 0
    if rank0:
        logger.info(f"[INFO] Training for {opt.trainer.max_epochs} epochs with {num_steps_per_epoch} steps per epoch")
        logger.info(f"[INFO] Evaluating every {opt.trainer.ckpt_every} interval with {num_steps_per_epoch_val} steps per epoch")

    if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.on_train_start()
    else: model.on_train_start()

    for epoch in range(int(opt.trainer.max_epochs)):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.current_epoch = epoch
        else: model.current_epoch = epoch
        train_loader_iter = iter(train_loader)
        if rank0: logger.info(f"Beginning epoch {epoch}...")
        with tqdm(train_loader, desc=f"Epoch {epoch}", disable=not rank0) as pbar:

            # 4.1 Training
            model.train()
            for input_dict in pbar:
                input_dict = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}

                optimizer.zero_grad()
                loss, log_dict = model(input_dict, mode='train')
                accelerator.backward(loss)
                optimizer.step()

                log_step += 1
                log_dict = accelerator.reduce(log_dict, 'mean')
                if rank0:
                    log_dict = {k: v.item() for k, v in log_dict.items()}
                    pbar.set_postfix(log_dict)
                    accelerator.log(log_dict, log_step)

            scheduler.step()

            # 4.2 Save checkpoint
            if rank0 and not (epoch % opt.trainer.save_img_every_epoch):
                # accelerator.save_state(os.path.join(exp_dir, f'epoch{epoch:04}.ckpt'))
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    else model.state_dict()
                torch.save(state_dict, os.path.join(opt.output_dir, 'ckpt', f'model_{epoch:04}.ckpt'))
                torch.save(optimizer.state_dict(), os.path.join(opt.output_dir, 'ckpt', f'optim_{epoch:04}.ckpt'))
                torch.save(scheduler.state_dict(), os.path.join(opt.output_dir, 'ckpt', f'sched_{epoch:04}.ckpt'))
                logger.info(f"Saved checkpoint at epoch {epoch} step {log_step} to {opt.output_dir}")

            # 4.3 Evaluation
            if opt.trainer.eval_every > 0 and (epoch % opt.trainer.eval_every == 0):
                logger.info(f"Evaluating on epoch {epoch} global_step {log_step}......")
                model.eval()
                with torch.inference_mode():
                    val_loader_iter = iter(val_loader)
                    val_size = torch.tensor(num_steps_per_epoch_val, device=accelerator.device)
                    sum_val_dict = defaultdict(float)
                    with trange(num_steps_per_epoch_val,
                                desc=f"Validation Epoch {epoch}",
                                disable=not rank0) as pbar_val:
                        for _ in pbar_val:
                            input_dict = next(val_loader_iter)
                            input_dict = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                            _, log_val_dict = model(input_dict, mode='val')
                            for k, v in log_val_dict.items(): sum_val_dict[k] += v
                            log_val_dict = {k: v.item() for k, v in log_val_dict.items()}
                            if rank0: pbar_val.set_postfix(log_val_dict)

                val_size = accelerator.reduce(val_size, 'sum')
                sum_val_dict = accelerator.reduce(dict(sum_val_dict), 'sum') 
                if rank0:
                    sum_val_dict = {k: v.item() / val_size for k, v in sum_val_dict.items()}
                    accelerator.log(sum_val_dict, log_step)


class FilteredIO:
    _supperss_new_line = False
    def __init__(self, stream: TextIO, pattern: str): self.stream, self.pattern = stream, pattern
    def write(self, s: str):
        if s == self.pattern: self._supperss_new_line = True
        elif self._supperss_new_line: self._supperss_new_line = False
        else: self.stream.write(s)
    def flush(self): self.stream.flush()
if __name__ == '__main__':
    f_out = FilteredIO(stdout, 'npu gs_sort kernel donot support, using cpu radix sort instead!')
    with redirect_stdout(f_out): main()