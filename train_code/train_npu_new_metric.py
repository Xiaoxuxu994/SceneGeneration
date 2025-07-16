# %%
import os
import numpy as np
import random
import time
import torch
import torch_npu
import torch.multiprocessing as mp
# mp.set_start_method('spawn') 
from torch.multiprocessing import Process
from torch_npu.contrib import transfer_to_npu
torch._C._jit_set_profiling_mode(False)
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
## colossalai
import colossalai
# from colossalai.booster import Booster
# from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
# from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import queue
import math
import functools
from src.registry import DATASETS

## opensora 
from opensora_local.acceleration.checkpoint import set_grad_checkpoint
from opensora_local.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)

from opensora_local.datasets import prepare_dataloader
from opensora_local.utils.ckpt_utils import create_logger, load, save, load_checkpoint
from opensora_local.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora_local.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, to_torch_dtype

## src code
from src.registry import MODELS, build_module
# from src.dataset.dataset_build_scannet_v6 import DatasetForScanNetV6
from src.utils.metric import occ_eval_metric_all,init_running_metrics

import gc
# print(os.environ['MASTER_PORT'])

TMP_DIR = '/mnt/nvme3/ljr/tmp'

def cosine_warmup_schedule(warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return lr_lambda

def cleanup_memory():
    torch_npu.npu.empty_cache()
    gc.collect() 

def log_memory_usage(info=''):
    print(f"当前显存占用: {torch_npu.npu.memory_allocated()/1024**2:.2f}MB / {torch_npu.npu.max_memory_allocated()/1024**2:.2f}MB, {info}, {dist.get_rank()}")


def log_metrics(writer, logger, metrics, step_count, global_step, optimizer, step, mode='train'):
    avg_metrics = {k: v / step_count for k, v in metrics.items()}

    # 分组：默认、每个scale级别（如16、32、64、128）
    # default_keys = ['loss', 'loss_occ', 'logit_mean', 'logit_max', 'logit_min',
    #                 'Precision_cl1', 'Recall_cl1', 'F1_cl1', 'IoU_cl1',
    #                 'Precision_cl0', 'Recall_cl0', 'F1_cl0', 'IoU_cl0',
    #                 'Precision_wavg', 'Recall_wavg', 'F1_wavg', 'IoU_wavg']

    default_keys = ['loss', 'loss_occ','IoU_cl1', 'IoU_cl0', 'IoU_wavg']
    level_groups = {}
    default_metrics = {}

    for k, v in avg_metrics.items():

        # all write in tensoboard
        writer.add_scalar(f"{mode}_{k}", v, global_step)

        # 仅保留前缀匹配的项
        if not any(k.startswith(prefix) for prefix in default_keys):
            continue

        # 分类为多层级还是默认
        if '_s' in k:
            *prefix, scale = k.split('_s')
            name = '_s'.join(prefix)
            scale = f's{scale}'
            if scale not in level_groups:
                level_groups[scale] = {}
            level_groups[scale][name] = v
        else:
            default_metrics[k] = v

    # 构建输出
    log_lines = [f"=={mode}_metrics (Step {step}, Global Step {global_step})=="]
    log_lines.append(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    for k, v in sorted(default_metrics.items()):
        log_lines.append(f"{k}: {v:.4f}")
        # writer.add_scalar(f"{mode}_{k}", v, global_step)

    for scale, group in level_groups.items():
        log_lines.append(f"-- Metrics for {scale} --")
        for k, v in sorted(group.items()):
            log_lines.append(f"{k}: {v:.4f}")
            # writer.add_scalar(f"{mode}_{k}_{scale}", v, global_step)

    logger.info("\n" + " | ".join(log_lines))
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)

def reduce_metrics(metrics_dict, device):
    if torch.distributed.is_initialized():
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                t = v.to(device)
            else:
                t = torch.tensor(v, dtype=torch.float32, device=device)
            metrics_dict[k] = all_reduce_mean(t)
    return metrics_dict

def validation_worker(shared_queue, stop_event):
    try:
        while True:
            try:
                msg, output_list, target_list, occ_threshold, running_metrics, scale_level, step, epoch, rank = shared_queue.get(timeout=0.5)
                if msg == "outputs":
                    for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                        scale = scale_level[idx]
                        metrics = occ_eval_metric_all(logit, target, occ_threshold=occ_threshold)
                        for name, score in metrics:
                            running_metrics[f'{name}_s{scale}'] += score
                        running_metrics[f'logit_mean_s{scale}'] += logit.mean().item()
                        running_metrics[f'logit_max_s{scale}'] += logit.max().item()
                        running_metrics[f'logit_min_s{scale}'] += logit.min().item()
                    if (step+1) % 500 == 0:
                        torch.save(running_metrics, os.path.join(TMP_DIR, f'res_{epoch}_{step}_{rank}.pth'))
                elif msg == "finished":
                    print('validation process finished')
                    break
            except queue.Empty:
                continue
            
    except Exception as e:
        print(f"Subprocess collapsed: {e}")
    finally:
        print("Subprocess Exit")
        stop_event.set()


def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    
    cfg = parse_configs(training=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.npu.manual_seed(cfg.seed)
    torch.npu.manual_seed_all(cfg.seed)
    if cfg.async_valid:
        manager = mp.Manager()
        shared_queue = manager.Queue()
        stop_event = mp.Event()
        cpu_proc = Process(
            target=validation_worker,
            args=(shared_queue, stop_event),
            daemon=True
        )
        cpu_proc.start()

    os.makedirs(TMP_DIR, exist_ok=True)
    
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.npu.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp32", "fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"
    
    # 2.1. colossalai init distributed training
    colossalai.launch_from_torch(backend='hccl')
    coordinator = DistCoordinator()
    device = get_current_device()
    if coordinator.is_master():
        print("====== cfg: ",cfg)
    else:
        print("====== exp_dir: ", exp_dir)
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")
        writer = create_tensorboard_writer(exp_dir)
    
    logger.info(f"Train cfg: {cfg}")
    
    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        # plugin = LowLevelZeroPlugin(
        #     stage=2,
        #     precision=cfg.dtype,
        #     initial_scale=2**16,
        #     max_norm=cfg.grad_clip,
        # )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        from opensora_local.acceleration.plugin import ZeroSeqParallelPlugin
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    # booster = Booster(plugin=plugin)
    
    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet
    scale_level = cfg.data_args['vs_res']
    # print(MODELS)
    train_dataset = build_module(cfg.train_dataset, DATASETS)
    val_dataset = build_module(cfg.val_dataset, DATASETS)

    train_dataloader = prepare_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )

    val_dataloader = prepare_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )

    logger.info(f"Train Dataset contains {len(train_dataset)} data: ({train_dataset.data_list_name})")
    logger.info(f"Val Dataset contains {len(val_dataset)} data: ({val_dataset.data_list_name})")

    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    model = build_module(cfg.model, MODELS)

    # 4.2. move to device
    model = model.to(device, dtype)

    # 4.3. load pretrain model 
    if cfg.load is not None:
        logger.info("Loading checkpoint......")
        load_checkpoint(model, cfg.load, save_as_pt=False, strict=False)  

    # 4.4. setup optimizer & lr_scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # cosinLR+warmup
    total_steps = cfg.epochs * len(train_dataloader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=cosine_warmup_schedule(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=cfg.min_lr_ratio
        )
    )
    # 4.5. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)

    # 4.6. set training mode
    model.train()

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    # model = DDP(model, device_ids=[device.index], bucket_cap_mb=10, gradient_as_bucket_view=True)
    model = DDP(model, device_ids=[device.index])
    
    torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(train_dataloader)
    num_steps_per_epoch_val = len(val_dataloader)
    logger.info("Boost model for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = start_step_eval = 0

    # 6.1. resume training
    if cfg.resume is not None:
        logger.info("Resuming checkpoint......")
        start_epoch, start_step, sampler_start_idx = load(None, model, optimizer, lr_scheduler, cfg.resume)
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")
    logger.info(f"Evaluating every {cfg.ckpt_every} interval with {num_steps_per_epoch_val} steps per epoch")
  
    train_dataloader.sampler.set_start_index(sampler_start_idx)
    val_dataloader.sampler.set_start_index(0)
    

    # =======================================================
    # 7. Model Info Investigate
    # =======================================================
    # 7.1. Print training parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"==> Training Parameter: {name}")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            logger.info(f"==> Fronzen Parameter: {name}")

    logger.info("============== model structure =========================")
    logger.info(model)
    logger.info("========================================================")
    
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # =======================================================
    # 8. Training loss and Eval metric Define
    # =======================================================
    # 8.1. Training loss related   
    # bceloss
    if cfg.BEC_pos_weight:
        pos_weight = 10
        occ_criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weight]).to(device, dtype))
    else:
        occ_criterion = torch.nn.BCEWithLogitsLoss()

    profiling = False

    if profiling:
        # time_fwd, time_bwd = [], []
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
        prof = torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU,
                                    torch_npu.profiler.ProfilerActivity.CPU],
                        with_stack=True,
                        record_shapes=False,
                        profile_memory=True,
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=0,
                                                                skip_first=0),
                        experimental_config=experimental_config,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profiling/train_encode_c2f_async/'))
        # prof.start()


    # =======================================================
    # 9. Training Loop
    # =======================================================
    for epoch in range(start_epoch, cfg.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        traindataloader_iter = iter(train_dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        running_metrics = init_running_metrics(levels=scale_level,
                                               class_keys=['cl1','cl0','wavg'],
                                               base_metrics=['Precision','Recall','F1','IoU'],
                                               include_per_level=True)
        log_step = 0

        with tqdm(range(start_step, num_steps_per_epoch),desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(), total=num_steps_per_epoch, initial=start_step,) as pbar:
            for step in pbar:
                # log_memory_usage('before forward')
                input_dict = next(traindataloader_iter)
                input_dict = {k: v[0].to(device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                model.train()

                if profiling:
                    if step == 140:
                        prof.start()

                # log_memory_usage('before forward 2')
                output_list, target_list = model.forward(input_dict)
                # log_memory_usage(f'after forward, step {step}')
                loss = torch.tensor(0).to(device, dtype)
                for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                    scale = scale_level[idx]
                    loss_occ = cfg.loss_weight_occ[idx] * occ_criterion(logit, target)
                    if (not torch.isnan(loss_occ)) and (not torch.isinf(loss_occ)):
                        loss += loss_occ 
                        running_metrics[f'loss_occ_s{scale}'] += loss_occ.item()                    
                
                optimizer.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"⚠️ Parameter without gradient: {name}")

                # exit(-1)
                optimizer.step()
                # log_memory_usage('After backward')
                if lr_scheduler is not None:
                    lr_scheduler.step()
                if profiling:
                    if step >= 140:
                        prof.step()
                        if step > 145:
                            prof.stop()
                            print('End profiling...')
                            exit(-1)

                ## evaluation
                running_metrics['loss'] += loss.item()
                # import time
                # torch.npu.synchronize()
                # a = time.time()
                log_step += 1
                global_step = epoch * num_steps_per_epoch + step
                if not cfg.async_valid:
                    if (global_step) % cfg.log_every == 0:
                        for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                            scale = scale_level[idx]
                            metrics = occ_eval_metric_all(logit, target, occ_threshold=cfg.occ_threshold)
                            for name, score in metrics:
                                running_metrics[f'{name}_s{scale}'] += score
                            running_metrics[f'logit_mean_s{scale}'] += logit.mean().item()
                            running_metrics[f'logit_max_s{scale}'] += logit.max().item()
                            running_metrics[f'logit_min_s{scale}'] += logit.min().item()
                        # torch.npu.synchronize()
                        # b = time.time()
                        # print(f'*****Rank {dist.get_rank()}, Time = {b - a:.4f} s')
                        ## all reduce mean
                        # print('*****')
                        running_metrics = reduce_metrics(running_metrics,device)

                else:
                    with torch.no_grad():
                        shared_queue.put(("outputs", [x.detach().cpu() for x in output_list], [x.detach().cpu() for x in target_list], cfg.occ_threshold, running_metrics, scale_level, step+epoch*num_steps_per_epoch, epoch, dist.get_rank()))
                

                ## Log to tensorboard
                if cfg.async_valid:
                    # gather validation result at the next step
                    if global_step == 0:
                        i = 0
                        while not os.path.exists(os.path.join(TMP_DIR, 'res_0_0_0.pth')):
                            print('Waiting for first evaluation' + '.' * i, flush=True)
                            time.sleep(0.1)
                        load_dir = f'{TMP_DIR}/res_0_0_'
                    else:
                        load_dir = f'{TMP_DIR}/res_{epoch}_{global_step-1}_'
                    world_size = coordinator.world_size
                    # Reduce Results
                    if coordinator.is_master() and (global_step) % cfg.log_every == 0:
                        while not os.path.exists(load_dir+'0.pth'):
                            time.sleep(0.1)
                        keys = torch.load(load_dir+'0.pth', map_location='cpu').keys()
                        running_metrics = {k:0 for k in keys}
                        for i in range(world_size):
                            load_path = load_dir + f'{i}.pth'
                            this_metrics = torch.load(load_path, map_location='cpu')
                            running_metrics = {
                                k: running_metrics[k]+v
                                for k,v in this_metrics.items()
                            }
                        running_metrics = {k:v/world_size for k,v in running_metrics.items()}


                if coordinator.is_master() and (global_step) % cfg.log_every == 0:
                    log_metrics(writer, logger, running_metrics, log_step, global_step, optimizer, step,mode='train')
                    running_metrics = {k: 0.0 for k in running_metrics}
                    log_step = 0

                # 清理释放
                del input_dict, output_list, target_list, loss, loss_occ
                cleanup_memory()

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step) % cfg.ckpt_every == 0:
                    save(None, model, optimizer, lr_scheduler, epoch, step, global_step, cfg.batch_size, coordinator, exp_dir)
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step} global_step {global_step} to {exp_dir}")

                    ### Eval Set
                    logger.info(f"Evaluating {epoch} step {step} global_step {global_step}......")
                    model.eval()
                    running_eval_metrics = init_running_metrics(levels=scale_level,
                                               class_keys=['cl1','cl0','wavg'],
                                               base_metrics=['Precision','Recall','F1','IoU'],
                                               include_per_level=True)
                    eval_log_step = 0
                    val_dataloader.sampler.set_epoch(epoch)
                    valdataloader_iter = iter(val_dataloader)
                    with tqdm(range(start_step_eval, num_steps_per_epoch_val),desc=f"Epoch {epoch}",
                            disable=not coordinator.is_master(),total=num_steps_per_epoch_val,initial=start_step_eval,) as pbar_val:
                        for _ in pbar_val:
                            input_dict = next(valdataloader_iter)
                            input_dict = {k: v[0].to(device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}

                            with torch.no_grad():
                                output_list, target_list = model.forward(input_dict)

                            eval_loss = torch.tensor(0).to(device, dtype)
                            for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                                scale = scale_level[idx]
                                eval_loss_occ = cfg.loss_weight_occ[idx] * occ_criterion(logit.detach(), target.detach())
                                if (not torch.isnan(eval_loss_occ)) and (not torch.isinf(eval_loss_occ)):
                                    eval_loss += eval_loss_occ 
                                    running_eval_metrics[f'loss_occ_s{scale}'] += eval_loss_occ.item()                                 


                            ## evaluation
                            running_eval_metrics['loss'] += eval_loss.item()
                            for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                                scale = scale_level[idx]
                                metrics = occ_eval_metric_all(logit, target, occ_threshold=cfg.occ_threshold)
                                for name, score in metrics:
                                    running_eval_metrics[f'{name}_s{scale}'] += score
                                running_eval_metrics[f'logit_mean_s{scale}'] += logit.mean().item()
                                running_eval_metrics[f'logit_max_s{scale}'] += logit.max().item()
                                running_eval_metrics[f'logit_min_s{scale}'] += logit.min().item()
                            eval_log_step += 1

                            # 清理本 step 显存
                            del input_dict, output_list, target_list,eval_loss, eval_loss_occ
                            cleanup_memory()

                    # all readuce mean         
                    running_eval_metrics = reduce_metrics(running_eval_metrics, device)    
                    if coordinator.is_master():
                        log_metrics(writer, logger, running_eval_metrics, eval_log_step, global_step, optimizer, step, mode='eval')
                        log_memory_usage()
                        cleanup_memory() 

                    val_dataloader.sampler.set_start_index(0)
                    start_step_eval = 0                       
            
        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        train_dataloader.sampler.set_start_index(0)
        start_step = 0
        torch.cuda.empty_cache()
        exit(-1)

    if cfg.async_valid:
        shared_queue.put(("finish", None, None, None, None, None, None, None, None))
        stop_event.wait()
        print('Validation Process finished')
        cpu_proc.join()


if __name__ == "__main__":
    main()
    
    
    
