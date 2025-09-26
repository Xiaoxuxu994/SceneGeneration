# %%
import os
import numpy as np
import random
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch._C._jit_set_profiling_mode(False)
# import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# import deepspeed
# import deepspeed.comm as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, InitProcessGroupKwargs
## colossalai
# import colossalai
# from colossalai.booster import Booster
# from colossalai.booster.plugin import LowLevelZeroPlugin
# from colossalai.cluster import DistCoordinator
# from colossalai.nn.optimizer import HybridAdam
# from colossalai.utils import get_current_device

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

## opensora 
from opensora_local.acceleration.checkpoint import set_grad_checkpoint
from opensora_local.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)

from opensora_local.datasets import prepare_dataloader
from opensora_local.utils.ckpt_utils import create_logger, deepspeed_load, deepspeed_save, load_checkpoint, save, load
from opensora_local.utils.config_utils import (
    create_experiment_workspace_all,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora_local.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, to_torch_dtype

## src code
from src.registry import MODELS, build_module, build_dataset_kwargs
from src.dataset.dataset_build_scannet_v6 import DatasetForScanNetV6DEBUGAUGSDF
from src.utils.metric import init_running_metrics, sdf_eval_metric_all_TM, sdf_eval_metric_all

import gc
import time
# print(os.environ['MASTER_PORT'])
from torch.utils.tensorboard import SummaryWriter

def softclip(logit, tau, type:str='sigmoid', alpha:float=1.0):
    if type == 'sigmoid':
        return tau * torch.sigmoid(logit * alpha)
    if type == 'sigmoid_center':
        return tau * torch.sigmoid((logit - tau/2) * alpha)
    if type == 'swish':
        swish = logit * torch.sigmoid(logit)
        return tau * torch.sigmoid(swish * alpha)
    if type == 'swish_center':
        swish = logit * torch.sigmoid(logit)
        return tau * torch.sigmoid((swish - tau/4) * alpha)
    if type == 'double_leaky_relu':
        neg_part = logit * alpha
        mid_part = logit
        pos_part = tau + (logit - tau)*alpha
        # Masks
        neg_mask = (logit < 0).float()
        mid_mask = ((logit >= 0) & (logit <= tau)).float()
        pos_mask = (logit > tau).float()
        # Combine all parts
        return neg_mask * neg_part + mid_mask * mid_part + pos_mask * pos_part

class StabilityMonitor:
    def __init__(self, model, log_dir="analysis/stability"):
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Module):
                continue

            module._name = name  # 保存模块名用于记录

            def forward_hook(mod, inp, out):
                stats = {}
                if isinstance(out, (tuple, list)):
                    for ind, val in enumerate(out):
                        if isinstance(val, torch.Tensor):
                            stats[f"min_{ind}"] = val.min().item()
                            stats[f"max_{ind}"] = val.max().item()
                            stats[f"mean_{ind}"] = val.mean().item()
                            stats[f"std_{ind}"] = val.std().item()
                elif isinstance(out, torch.Tensor):
                    stats = {
                        "min": out.min().item(),
                        "max": out.max().item(),
                        "mean": out.mean().item(),
                        "std": out.std().item()
                    }
                mod._forward_stats = stats

            def backward_hook(mod, grad_input, grad_output):
                stats = {}
                if isinstance(grad_output, (tuple, list)):
                    for ind, val in enumerate(grad_output):
                        if isinstance(val, torch.Tensor):
                            stats[f"min_{ind}"] = val.min().item()
                            stats[f"max_{ind}"] = val.max().item()
                            stats[f"mean_{ind}"] = val.mean().item()
                            stats[f"std_{ind}"] = val.std().item()
                            stats[f"norm_{ind}"] = val.norm().item()
                elif isinstance(grad_output, torch.Tensor):
                    stats = {
                        "min": grad_output.min().item(),
                        "max": grad_output.max().item(),
                        "mean": grad_output.mean().item(),
                        "std": grad_output.std().item(),
                        "norm": grad_output.norm().item()
                    }
                mod._backward_stats = stats

            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

    def log(self, step):
        for module in self.model.modules():
            if hasattr(module, "_forward_stats"):
                for key, value in module._forward_stats.items():
                    self.writer.add_scalar(f"{module._name}/forward_{key}", value, step)

            if hasattr(module, "_backward_stats"):
                for key, value in module._backward_stats.items():
                    self.writer.add_scalar(f"{module._name}/backward_{key}", value, step)

    def close(self):
        self.writer.close()

class GradNormHook:
    def __init__(self, global_step_fn=None):
        """
        初始化 GradNorm Hook 管理器

        Args:
            global_step_fn (callable): 返回 global_step 的函数（可选）
        """
        self.global_step_fn = global_step_fn
        self._norms = []

    def register_hooks(self, model):
        """
        注册 hook 到模型所有参数
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook_fn(grad):
            norm = grad.norm().item()
            self._norms.append(norm)
        return hook_fn

    def compute(self):
        """
        计算 mean grad norm
        """
        if not self._norms:
            return 0.0
        mean_norm = sum(self._norms) / len(self._norms)
        self._norms.clear()
        return mean_norm
    
def get_current_device():
    if torch.npu.is_available():
        dev_id = torch.npu.current_device()
        return torch.device(f"npu:{dev_id}")
    else:
        return torch.device("cpu")
    
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

def log_memory_usage():
    print(f"当前显存占用: {torch_npu.npu.memory_allocated()/1024**2:.2f}MB / {torch_npu.npu.max_memory_allocated()/1024**2:.2f}MB")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
             torch.nn.init.zeros_(m.bias)

def log_metrics(writer, logger, metrics, step_count, global_step, optimizer, step, mode='train', index=0):
    avg_metrics = {k: v / step_count for k, v in metrics.items()}

    # 分组：默认、每个scale级别（如16、32、64、128）
    # default_keys = ['loss', 'loss_occ', 'logit_mean', 'logit_max', 'logit_min',
    #                 'Precision_cl1', 'Recall_cl1', 'F1_cl1', 'IoU_cl1',
    #                 'Precision_cl0', 'Recall_cl0', 'F1_cl0', 'IoU_cl0',
    #                 'Precision_wavg', 'Recall_wavg', 'F1_wavg', 'IoU_wavg']

    default_keys = ['loss', 'loss_kl','loss_sdf','grad_norm','IoU_cl1', 'IoU_cl0', 'IoU_wavg','SDF_L1','SDF_L2']
    level_groups = {}
    default_metrics = {}

    for k, v in avg_metrics.items():

        # all write in tensoboard
        writer.add_scalar(f"{mode}_{k}_{index}", v, global_step)

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
    
    # 2.1. colossalai init distributed training
    # deepspeed.init_distributed(dist_backend='hccl')
    # project_config = ProjectConfiguration(project_dir=exp_dir, logging_dir=os.path.join(exp_dir, 'log'))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(
        backend="hccl"
    )
    # mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
    accelerator = Accelerator(
        # project_config=project_config,
        # log_with='tensorboard',
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        # kwargs_handlers=[ddp_kwargs],
    )

    # 2.2. init logger, tensorboard
    exp_dir, model_dir, monitor_dir = create_experiment_workspace_all(cfg, accelerator.is_main_process)
    if accelerator.is_main_process: # only makedir on global_rank0 process, also logger and tensorboard writer
        save_training_config(cfg._cfg_dict, exp_dir)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")
        writer = create_tensorboard_writer(exp_dir)
    else:
        logger = create_logger(None)
    logger.info(f"Train cfg: {cfg}")
    
    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.npu.is_available(), "Training currently requires at least one NPU."
    assert cfg.dtype in ["fp32", "fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # print(accelerator.process_index, accelerator.num_processes, '*****')
    # exit(-1)
    # colossalai.launch_from_torch(backend='hccl')
    # coordinator = DistCoordinator()
    # deepspeed.initialize()
    # accelerator.local_process_index: local rank
    # dist.get_rank: global rank
    # e.g., 2 nodes, 16 gpus, node_rank = 1
    # dist.get_rank() = 9, accelerator.local_process_index=1.
    device = torch.device(f'npu:{accelerator.local_process_index}')
    if accelerator.local_main_process_first() == 0:
        print("====== cfg: ",cfg)
    else:
        print("====== exp_dir: ", exp_dir)
    dtype = to_torch_dtype(cfg.dtype)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet
    scale_level = cfg.vs_res
    train_dataset = DatasetForScanNetV6DEBUGAUGSDF(
        data_root=cfg.data_root,
        data_list_name=cfg.train_data_list_name,
        **build_dataset_kwargs(cfg.train_data_args)
    )
    val_dataset_list = [DatasetForScanNetV6DEBUGAUGSDF(
        data_root=cfg.data_root,
        data_list_name=val_data_list_name,
        **build_dataset_kwargs(cfg.val_data_args)
    ) for val_data_list_name in cfg.val_data_list_name]

    train_dataloader = prepare_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        world_size=accelerator.num_processes,
        rank=accelerator.process_index
    )

    val_dataloader_list = [prepare_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        world_size=accelerator.num_processes,
        rank=accelerator.process_index
    ) for val_dataset in val_dataset_list]

    logger.info(f"Train Dataset contains {len(train_dataset)} data: ({cfg.train_data_list_name})")
    for val_dataset, val_dataset_name in zip(val_dataset_list, cfg.val_data_list_name):
        logger.info(f"Val Dataset contains {len(val_dataset)} data: ({val_dataset_name})")

    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    model = build_module(cfg.model, MODELS)
    if cfg.init_with_xavier:
        model.apply(init_weights)

    # 4.2. move to device
    model = model.to(device, dtype)

    # 4.2.5 register hook class to model, output to each rank dir
    if cfg.monitor:
        monitor_rank_dir = f"{monitor_dir}/rank{dist.get_rank()}"
        os.makedirs(monitor_rank_dir, exist_ok=True)
        monitor = StabilityMonitor(model, log_dir=monitor_rank_dir)

    # 4.3. load pretrain model  cfg.load=/exp_name/model/model.pt
    if cfg.load is not None:
        logger.info("Loading checkpoint......")
        load_checkpoint(model, cfg.load, save_as_pt=False, strict=False)
    total_steps = cfg.epochs * len(train_dataloader)
    # print('len(train_dataloader): ',len(train_dataloader))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    # Test DeepSpeed

    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # print(device, '***********************')
    # model, optimizer, _, _ = deepspeed.initialize(
    #     model=model,
    #     model_parameters=parameters,
    #     config_params=deepspeed_config
    # )

    # 4.4. setup optimizer & lr_scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # cosinLR+warmup
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

    # 4.6. choose metric calc func
    calc_metric = sdf_eval_metric_all if cfg.use_cpu_metric else sdf_eval_metric_all_TM

    # 4.7. set training mode
    model.train()
    # print(train_dataloader.sampler, '************Before')
    # if accelerator.process_index == 0:
    #     import pdb; pdb.set_trace()
    # else:
    #     import time; time.sleep(12000)
    val_dataloader_list.append(lr_scheduler)
    model, optimizer, train_dataloader, val_dataloader_list = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader_list)
    lr_scheduler = val_dataloader_list.pop()
    # train_dataloader = accelerator.prepare_data_loader(train_dataloader, device_placement=False)
    # val_dataloader = accelerator.prepare_data_loader(val_dataloader, device_placement=False)
    # print(train_dataloader.sampler, len(train_dataloader), '*************AFter')

    # exit(-1)
    
    

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    # torch.set_default_dtype(dtype)
    # model = DDP(model, device_ids=[device.index])
    
    torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(train_dataloader)
    num_steps_per_epoch_val_list = [len(val_dataloader) for val_dataloader in val_dataloader_list]
    logger.info("Boost model for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    
    start_epoch = start_step = log_step = sampler_start_idx = start_step_eval = 0

    # 6.1. resume training
    if cfg.resume is not None:
        logger.info("Resuming checkpoint......") # cfg.resume = /exp_name/optimizer
        last_epoch, last_step, sampler_start_idx = deepspeed_load(model, cfg.resume)
        logger.info(f"Loaded checkpoint {cfg.resume} at epoch {last_epoch} step {last_step}")
        last_global_step = last_epoch * num_steps_per_epoch + last_step
        lr_scheduler.load_state_dict(torch.load(os.path.join(cfg.resume, f'global_step{last_global_step + 1}/lr_scheduler.pt'))) 
        start_step = last_step + 1 if last_step < num_steps_per_epoch - 1 else 0
        start_epoch = last_epoch if last_step < num_steps_per_epoch - 1 else last_epoch + 1
        # start_epoch, start_step, sampler_start_idx = load(None, model, optimizer, lr_scheduler, cfg.resume)    
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")
    logger.info(f"Evaluating every {cfg.ckpt_every} interval with {num_steps_per_epoch_val_list} steps per epoch")
  
    # train_dataloader.sampler.set_start_index(sampler_start_idx)
    # val_dataloader.sampler.set_start_index(0)
    

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

    # sdf loss
    if cfg.enable_sdf_loss:
        if cfg.sdf_loss_type == 'l1':
            sdf_criterion = torch.nn.L1Loss()
        elif cfg.sdf_loss_type == 'mse':
            sdf_criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown sdf_loss_type: {cfg.sdf_loss_type}")
        
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
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profiling/train_sdf_256/'))

    # =======================================================
    # 9. Training Loop
    # =======================================================
    all_step = cfg.epochs * num_steps_per_epoch
    grad_hook = GradNormHook(global_step_fn=lambda: global_step)
    grad_hook.register_hooks(model)  # 注册 hook 到模型参数
    for epoch in range(start_epoch, cfg.epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        traindataloader_iter = iter(train_dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        if epoch == cfg.gt_warmup_epochs:
            logger.info(f"Switched to standard coarse-to-fine training at epoch {epoch}.")

        running_metrics = init_running_metrics(levels=scale_level,
                                               class_keys=['cl1','cl0','wavg'],
                                               base_metrics=['Precision','Recall','F1','IoU'],
                                               include_per_level=True)
        log_step = 0

        with tqdm(range(start_step, num_steps_per_epoch),desc=f"Epoch {epoch}",
            disable=not accelerator.local_process_index == 0, total=num_steps_per_epoch, initial=start_step,) as pbar:
            for step in pbar:
                input_dict = next(traindataloader_iter)
                input_dict = {k: v[0].to(device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                model.train()
                
                # core_model = model.module if hasattr(model, 'module') else model
                # if epoch < cfg.gt_warmup_epochs:
                #     core_model.set_mode('gt')
                # else:
                #     core_model.set_mode('coarse_to_fine')
                if profiling:
                    if step == 140:
                        prof.start()

                loss_kl, output_list, target_list = model(input_dict,vs=scale_level[-1])
                # print('**********', input_dict["coord"].shape, f'Rank {accelerator.local_main_process_first()}')
                

                loss = torch.tensor(0).to(device, dtype)
                for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                    scale = scale_level[idx]
                    if cfg.enable_occ_loss:
                        loss_occ = cfg.loss_weight_occ[idx] * occ_criterion(logit, target)
                        if (not torch.isnan(loss_occ)) and (not torch.isinf(loss_occ)):
                            loss += loss_occ 
                            running_metrics[f'loss_occ_s{scale}'] += loss_occ.item()    
                            running_metrics['loss_occ'] += loss_occ.item()  # accumulate all occ loss across all level                    

                    ## enable_sdf_loss
                    if cfg.enable_sdf_loss:
                        if cfg.truncatedist:
                            if not cfg.softclip or epoch >= cfg.epochs * cfg.epoch_ratio:
                                loss_sdf = cfg.loss_weight_SDF[idx] * \
                                (sdf_criterion(
                                    torch.clip(logit, 0.0, cfg.truncatedist), 
                                    torch.clip(target, 0.0, cfg.truncatedist))                              
                                )
                            else:
                                loss_sdf = cfg.loss_weight_SDF[idx] * \
                                (sdf_criterion(
                                    softclip(logit, cfg.truncatedist, type=cfg.softcliptype, alpha=cfg.softclipalpha),
                                    torch.clip(target, 0.0, cfg.truncatedist))
                                )
                        else:
                            loss_sdf = cfg.loss_weight_SDF[idx] * sdf_criterion(logit, target)
                        if (not torch.isnan(loss_sdf)) and (not torch.isinf(loss_sdf)):
                            loss +=  loss_sdf
                            running_metrics[f'loss_sdf_s{scale}'] += loss_sdf.item()
                            running_metrics['loss_sdf'] += loss_sdf.item()

                # kl loss
                if (not torch.isnan(loss_kl)) and (not torch.isinf(loss_kl)):
                    loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
                    loss += cfg.loss_weight_KL * loss_kl 
                    running_metrics['loss_kl'] += loss_kl.item()   

                optimizer.zero_grad()
                # loss.backward()

                # model.backward(loss)
                accelerator.backward(loss)
                # model.step()

                # log gradnorm
                running_metrics['grad_norm']  += grad_hook.compute()

                optimizer.step()
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
                for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                    scale = scale_level[idx]
                    # metrics = occ_eval_metric_all(logit, target, occ_threshold=cfg.occ_threshold)
                    metrics = calc_metric(logit, target, sdf_threshold=cfg.sdf_threshold, mask=None, truncatedist = cfg.truncatedist,compute_surface_metrics=False)
                    for name, score in metrics:
                        running_metrics[f'{name}_s{scale}'] += score
                    # running_metrics[f'logit_mean_s{scale}'] += logit.mean().item()
                    # running_metrics[f'logit_max_s{scale}'] += logit.max().item()
                    # running_metrics[f'logit_min_s{scale}'] += logit.min().item()

                ## all reduce mean
                running_metrics = reduce_metrics(running_metrics,device)                
                log_step += 1
                global_step = epoch * num_steps_per_epoch + step

                ## Log to tensorboard
                if (global_step) % cfg.log_every == 0 or global_step + 1 == all_step:
                    if accelerator.is_main_process: # 只在global_rank = 0上写入log和writer
                        log_metrics(writer, logger, running_metrics, log_step, global_step, optimizer, step,mode='train')
                    running_metrics = {k: 0.0 for k in running_metrics} # every log_every steps, clear out running_metrics
                    log_step = 0
                
                if cfg.monitor and cfg.monitor_every > 0 and global_step % cfg.monitor_every == 0:
                    monitor.log(global_step)

                # 清理释放
                # del input_dict, output_list, target_list, loss, loss_occ,loss_kl
                del input_dict, output_list, target_list, loss, loss_sdf,loss_kl

                # Save checkpoint
                if cfg.ckpt_every > 0 and ((global_step + 1) % cfg.ckpt_every == 0 or global_step + 1 == all_step):
                    # save(None, model, optimizer, lr_scheduler, epoch, step, global_step, cfg.batch_size, exp_dir)
                    deepspeed_save(model, epoch, step, cfg.batch_size, global_step, model_dir)
                    if accelerator.is_main_process: # only save lr_scheduler on global_rank0
                        torch.save(lr_scheduler.state_dict(), os.path.join(model_dir, f'global_step{global_step + 1}/lr_scheduler.pt'))
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step} global_step {global_step} to {model_dir}")
                
                if cfg.eval_every > 0 and ((global_step + 1) % cfg.eval_every == 0 or global_step + 1 == all_step):
                    ### Eval Set
                    logger.info(f"Evaluating {epoch} step {step} global_step {global_step}......")
                    model.eval()
                    for val_index, (val_dataloader, num_steps_per_epoch_val) in enumerate(zip(val_dataloader_list, num_steps_per_epoch_val_list)):
                        running_eval_metrics = init_running_metrics(levels=scale_level,
                                                class_keys=['cl1','cl0','wavg'],
                                                base_metrics=['Precision','Recall','F1','IoU'],
                                                include_per_level=True)
                        eval_log_step = 0
                        # val_dataloader.sampler.set_epoch(epoch)
                        valdataloader_iter = iter(val_dataloader)
                        with tqdm(range(start_step_eval, num_steps_per_epoch_val),desc=f"Epoch {epoch}",
                                disable=not accelerator.local_process_index==0,total=num_steps_per_epoch_val,initial=start_step_eval,) as pbar_val:
                            for _ in pbar_val:
                                input_dict = next(valdataloader_iter)
                                input_dict = {k: v[0].to(device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}

                                with torch.no_grad():
                                    # core_model = model.module if hasattr(model, 'module') else model
                                    # if epoch < cfg.gt_warmup_epochs:
                                    #     core_model.set_mode('gt')
                                    # else:
                                    #     core_model.set_mode('coarse_to_fine')
                                    eval_loss_kl, output_list, target_list = model(input_dict,vs=scale_level[-1])

                                eval_loss = torch.tensor(0).to(device, dtype)
                                for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                                    scale = scale_level[idx]
                                    if cfg.enable_occ_loss:
                                        eval_loss_occ = cfg.loss_weight_occ[idx] * occ_criterion(logit.detach(), target.detach())
                                        if (not torch.isnan(eval_loss_occ)) and (not torch.isinf(eval_loss_occ)):
                                            eval_loss += eval_loss_occ 
                                            running_eval_metrics[f'loss_occ_s{scale}'] += eval_loss_occ.item()      
                                            running_eval_metrics['loss_occ'] += eval_loss_occ.item()  # accumulate all occ loss across all level                              

                                    ## enable_sdf_loss
                                    if cfg.enable_sdf_loss:
                                        if cfg.truncatedist:
                                            if not cfg.softclip:
                                                eval_loss_sdf = cfg.loss_weight_SDF[idx] * \
                                                (sdf_criterion(
                                                    torch.clip(logit, 0.0, cfg.truncatedist), 
                                                    torch.clip(target, 0.0, cfg.truncatedist))                            
                                                )
                                            else:
                                                eval_loss_sdf = cfg.loss_weight_SDF[idx] * \
                                                (sdf_criterion(
                                                    softclip(logit, cfg.truncatedist, type=cfg.softcliptype, alpha=cfg.softclipalpha),
                                                    torch.clip(target, 0.0, cfg.truncatedist))
                                                )
                                        else:
                                            eval_loss_sdf = cfg.loss_weight_SDF[idx] * sdf_criterion(logit, target)

                                        if (not torch.isnan(eval_loss_sdf)) and (not torch.isinf(eval_loss_sdf)):
                                            eval_loss +=  eval_loss_sdf
                                            running_eval_metrics[f'loss_sdf_s{scale}'] += eval_loss_sdf.item()
                                            running_eval_metrics['loss_sdf'] += eval_loss_sdf.item()

                                # kl loss
                                if (not torch.isnan(eval_loss_kl)) and (not torch.isinf(eval_loss_kl)):
                                    eval_loss_kl = torch.sum(eval_loss_kl) / eval_loss_kl.shape[0]
                                    eval_loss += cfg.loss_weight_KL * eval_loss_kl 
                                    running_eval_metrics['loss_kl'] += eval_loss_kl.item()

                                ## evaluation
                                running_eval_metrics['loss'] += eval_loss.item()
                                for idx,(logit, target) in enumerate(zip(output_list,target_list)):
                                    scale = scale_level[idx]
                                    # metrics = occ_eval_metric_all(logit, target, occ_threshold=cfg.occ_threshold)
                                    metrics = calc_metric(logit, target, sdf_threshold=cfg.sdf_threshold, mask=None, truncatedist = cfg.truncatedist, compute_surface_metrics=False) 
                                    for name, score in metrics:
                                        running_eval_metrics[f'{name}_s{scale}'] += score
                                    # running_eval_metrics[f'logit_mean_s{scale}'] += logit.mean().item()
                                    # running_eval_metrics[f'logit_max_s{scale}'] += logit.max().item()
                                    # running_eval_metrics[f'logit_min_s{scale}'] += logit.min().item()
                                eval_log_step += 1

                                # 清理本 step 显存
                                # del input_dict, output_list, target_list,eval_loss, eval_loss_occ, eval_loss_kl
                                del input_dict, output_list, target_list,eval_loss, eval_loss_sdf, eval_loss_kl

                        # all readuce mean         
                        running_eval_metrics = reduce_metrics(running_eval_metrics, device)    
                        if dist.get_rank() == 0:
                            log_metrics(writer, logger, running_eval_metrics, eval_log_step, global_step, optimizer, step, mode='eval', index=val_index)
                            log_memory_usage()

                    # val_dataloader.sampler.set_start_index(0)
                    start_step_eval = 0                       
            
        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        # train_dataloader.sampler.set_start_index(0)
        start_step = 0
        torch.cuda.empty_cache()
        # exit(-1)


if __name__ == "__main__":
    main()
    
    
    
