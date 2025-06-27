# %%
import os
import datetime
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch._C._jit_set_profiling_mode(False)
import torch.distributed as dist
from torch.optim import AdamW
from torch_npu.npu.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm

## opensora 
from opensora_local.acceleration.checkpoint import set_grad_checkpoint
from opensora_local.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)

from opensora_local.datasets import prepare_dataloader
from opensora_local.utils.logger_utils import (
    create_logger,
    create_tensorboard_writer,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    extend_seed_all,
)
from opensora_local.utils.config_utils import (
    create_experiment_workspace,
    parse_configs,
    save_training_config,
)
from opensora_local.utils.misc import format_numel_str,  to_torch_dtype

## src code
from src.registry import MODELS, DATASETS, build_module
from src.utils.savepcd import save_inout_pc
from src.utils.metric import occ_eval_metric
from src.utils.stat import register_activation_hooks
from src.dataset.prefetcher import Prefetcher, AsyncSamplerProfiler

def main():
    rank = int(os.environ["LOCAL_RANK"])
    # args loading and write to experiment dir
    cfg = parse_configs(training=True)
    seed_everything(cfg.seed)
    if cfg.resume is not None and not os.path.exists(cfg.resume):
        cfg.resume =  None
        print("[warning] continue training but resume ckpt file not exist")
    if rank == 0:
        print("====== cfg: ",cfg)
        exp_name, exp_dir = create_experiment_workspace(cfg)
        save_training_config(cfg._cfg_dict, exp_dir)
        print("====== exp_dir: ", exp_dir)
    else:
        exp_dir = None

    # multi npu setting and device set
    n_npus = torch_npu.npu.device_count()
    assert n_npus >= 2, f"Requires at least 2 NPUs to run, but got {n_npus}"
    dist.init_process_group("hccl")
    torch.npu.set_device(rank)

    # random seed set
    
    assert cfg.dtype in ["fp32", "fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"
    dtype = to_torch_dtype(cfg.dtype)
    
    # init logger, tensorboard
    logger = create_logger(exp_dir)
    logger.info(f"Experiment directory created at {exp_dir}")
    writer = create_tensorboard_writer(exp_dir)
    logger.info(f"Train cfg: {cfg}")
    
    # init grad scaler
    if cfg.enable_scaler:
        scaler = GradScaler(init_scale=1 if dtype == torch.float32 else 2**16) 
    else:
        scaler = None
    
    # init data parallel group
    set_data_parallel_group(dist.group.WORLD)

    # build DatasetForScanNetV2/V3/V4
    train_dataset = build_module(cfg.train_dataset, DATASETS)
    val_dataset = build_module(cfg.val_dataset, DATASETS)

    # build dataloader
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
        batch_size=1,#cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    prefetcher_train = Prefetcher(train_dataloader, torch.device(rank), profiler=AsyncSamplerProfiler(), policy=train_dataset.policy)
    logger.info(f"Train Dataset contains {len(train_dataset)} data: ({cfg.train_dataset.data_list_name})")
    logger.info(f"Val Dataset contains {len(val_dataset)} data: ({cfg.val_dataset.data_list_name})")
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # build model and move to device
    model = build_module(cfg.model, MODELS)
    model = model.to(rank)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cos_total_iter, eta_min=cfg.cos_eta_min)
    model.train()

    # grad_checkpoint setting before ddp warp
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model, gc_step=cfg.grad_checkpoint_step)

    # ddp warp
    torch.set_default_dtype(dtype)
    model = DDP(model, device_ids=[rank])  #, find_unused_parameters=True)
    torch.set_default_dtype(torch.float32)
    # hooks = register_activation_hooks(model.module, rank, logger)
    
    # start training
    start_epoch = start_step = log_step = sampler_start_idx = start_step_eval = 0

    # resume training
    if cfg.resume is not None:
        logger.info("Resuming checkpoint......")
        start_epoch, start_step, sampler_start_idx = load_checkpoint(model, optimizer, lr_scheduler, scaler, cfg.resume)
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    
    # train and val steps
    num_steps_per_epoch = len(train_dataloader)
    num_steps_per_epoch_val = len(val_dataloader)
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")
    logger.info(f"Evaluating every {cfg.ckpt_every} interval with {num_steps_per_epoch_val} steps per epoch")
  
    train_dataloader.sampler.set_start_index(sampler_start_idx)
    val_dataloader.sampler.set_start_index(0)
    
    # Print training parameters
    if rank == 0:
        num_params = 0
        num_params_trainable = 0
        for name, param in model.named_parameters():
            print(f"{name}: shape={param.shape}, dtype={param.dtype}")
            num_params += param.numel()
            if param.requires_grad:
                num_params_trainable += param.numel()
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"==> Training Parameter: {name}")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                logger.info(f"==> Fronzen Parameter: {name}")
        logger.info("============== model structure =========================")
        logger.info(model)
        logger.info("========================================================")
        logger.info(
            f"Trainable model params: {format_numel_str(num_params_trainable)}, Total model params: {format_numel_str(num_params)}"
        )

    # train loss setting
    running_loss, running_loss_occ, running_loss_kl = 0.0, 0.0, 0.0

    # bceloss
    occ_criterion = torch.nn.BCEWithLogitsLoss()

    # eval metric
    running_iou, running_precision, running_recall, running_f1 = 0.0,0.0,0.0,0.0

    # train loop
    for epoch in range(start_epoch, cfg.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        traindataloader_iter = iter(train_dataloader)
        # prefetcher_train.reset(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # input_dict = prefetcher_train.next()

        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not rank==0,
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                input_dict = next(traindataloader_iter)
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key][0].to(rank)

                model.train()
                if scaler is not None:
                    with autocast(dtype=dtype):
                        outputs = model.forward(input_dict)
                else:
                    outputs = model.forward(input_dict)

                # calculate final loss
                loss = torch.tensor(0).to(rank, dtype)

                # occ loss
                pc_query_label = input_dict['pc_query_label'].unsqueeze(0)                
                output_logit = outputs['logits']

                loss_occ = cfg.loss_weight_occ * occ_criterion(output_logit, pc_query_label)
                running_loss_occ += loss_occ.item()

                # kl loss (may be None)
                loss_kl = outputs['kl']
                if loss_kl is not None:
                    loss_kl = cfg.loss_weight_kl * torch.sum(loss_kl) / loss_kl.shape[0]
                    running_loss_kl += loss_kl.item()
                
                if (not torch.isnan(loss_occ)) and (not torch.isinf(loss_occ)):
                    loss += loss_occ
                if (loss_kl is not None) and (not torch.isnan(loss_kl)) and (not torch.isinf(loss_kl)):
                    loss += loss_kl

                if scaler is not None:
                    # update model parameter with Scaler
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    next_batch = prefetcher_train.next()
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # update without scaler
                    optimizer.zero_grad()
                    loss.backward()
                    next_batch = prefetcher_train.next()
                    optimizer.step()
                    grad_norm = 0

                 # update scheduler after each iter
                if lr_scheduler is not None:
                    lr_scheduler.step() 

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                running_loss += loss.item()

                ## cal metric
                precision_score, recall_score, f1_score, iou_score = occ_eval_metric(cfg.eval_metric_list, output_logit, pc_query_label, cfg.occ_threshold)
                running_precision += (precision_score.mean()).item()
                running_recall += (recall_score.mean()).item()
                running_f1 += (f1_score.mean()).item()
                running_iou += (iou_score.mean()).item()  

                global_step = epoch * num_steps_per_epoch + step
                log_step += 1

                # Log to tensorboard
                if rank == 0 and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    avg_loss_occ = running_loss_occ / log_step
                    avg_loss_kl = running_loss_kl / log_step
                    avg_occ_precision = running_precision / log_step
                    avg_occ_recall = running_recall / log_step
                    avg_occ_f1 = running_f1 / log_step
                    avg_occ_iou = running_iou / log_step

                    logger.info(f"\n loss: {avg_loss}, avg_loss_occ: {avg_loss_occ}, avg_loss_kl: {avg_loss_kl}, avg_occ_precision: {avg_occ_precision}, avg_occ_recall: {avg_occ_recall}, avg_occ_f1: {avg_occ_f1}, avg_occ_iou: {avg_occ_iou}, step: {step}, global_step: {global_step}, lr: {optimizer.param_groups[0]['lr']}, grad_norm: {grad_norm}\n")
                    if scaler is not None:
                        logger.info(f"\n scale: {scaler.get_scale()}\n")
                    pbar.set_postfix({"loss": avg_loss,"avg_loss_occ": avg_loss_occ, "avg_loss_kl": avg_loss_kl, "avg_occ_precision": avg_occ_precision, "avg_occ_recall": avg_occ_recall, "avg_occ_f1": avg_occ_f1, "avg_occ_iou": avg_occ_iou, "step": step, "global_step": global_step, "lr": optimizer.param_groups[0]['lr'], "grad_norm": grad_norm})
                    pbar.set_postfix_str('')

                    running_loss, running_loss_occ, running_loss_kl = 0.0,0.0,0.0
                    running_iou, running_precision, running_recall, running_f1 = 0.0,0.0,0.0,0.0
                    log_step = 0

                    writer.add_scalar("loss", loss.item(), global_step)
                    writer.add_scalar("loss_occ", loss_occ.item(), global_step)
                    if loss_kl is not None:
                        writer.add_scalar("loss_kl", loss_kl.item(), global_step)
                    writer.add_scalar("train_precision_score", precision_score.mean().item(), global_step)
                    writer.add_scalar("train_recall_score", recall_score.mean().item(), global_step)
                    writer.add_scalar("train_f1_score", f1_score.mean().item(), global_step)
                    writer.add_scalar("train_iou_score", iou_score.mean().item(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)
                
                ### Save checkpoint
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    save_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, step, global_step, cfg.batch_size, exp_dir)
                    logger.info(f'-------------------------------------------------------------------------------')
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step} global_step {global_step} to {exp_dir}"
                    )
                    logger.info(f'-------------------------------------------------------------------------------')
                    if rank == 0:
                        inout_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step}", "input_output_results/train")
                        save_inout_pc(inout_dir, input_dict, output_logit)

                ### Eval Set
                if cfg.enable_occ_eval and cfg.eval_every > 0 and (global_step + 1) % cfg.eval_every == 0:
                    logger.info(f'-------------------------------------------------------------------------------')
                    logger.info(
                        f"Evaluating {epoch} step {step} global_step {global_step}......"
                    )
                    logger.info(f'-------------------------------------------------------------------------------')
                    eval_log_step = 0
                    running_eval_loss, running_eval_loss_occ, running_eval_loss_kl = 0.0, 0.0, 0.0
                    running_eval_precision, running_eval_recall, running_eval_f1, running_eval_iou = 0.0, 0.0, 0.0, 0.0
                    val_dataloader.sampler.set_epoch(epoch)
                    valdataloader_iter = iter(val_dataloader)
                    with tqdm(
                            range(start_step_eval, num_steps_per_epoch_val),
                            desc=f"Epoch {epoch}",
                            disable=not rank == 0,
                            total=num_steps_per_epoch_val,
                            initial=start_step_eval,
                        ) as pbar_val:
                        for _ in pbar_val:
                            input_dict_val = next(valdataloader_iter)
                            for key in input_dict_val.keys():
                                if isinstance(input_dict_val[key], torch.Tensor):
                                    input_dict_val[key] = input_dict_val[key][0].to(rank)

                            model.eval()
                            with torch.no_grad():
                                outputs = model.forward(input_dict_val)
                            
                            # calculate final loss
                            eval_loss = torch.tensor(0).to(rank, dtype)
                            # oss loss
                            pc_query_label = input_dict_val['pc_query_label'].unsqueeze(0)
                            output_logit = outputs['logits']
                            eval_loss_occ = cfg.loss_weight_occ * occ_criterion(output_logit.detach(), pc_query_label.detach())
                            running_eval_loss_occ += eval_loss_occ.item()
                            # kl loss (may be None)
                            eval_loss_kl = outputs['kl']
                            if eval_loss_kl is not None:
                                eval_loss_kl = cfg.loss_weight_kl * torch.sum(eval_loss_kl) / eval_loss_kl.shape[0]
                                running_eval_loss_kl += eval_loss_kl.item()
                            if (not torch.isnan(eval_loss_occ)) and (not torch.isinf(eval_loss_occ)):
                                eval_loss = eval_loss_occ
                            if (eval_loss_kl is not None) and (not torch.isnan(eval_loss_kl)) and (not torch.isinf(eval_loss_kl)):
                                eval_loss += eval_loss_kl

                            eval_loss_tensor = eval_loss.detach().clone()
                            dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                            running_eval_loss += eval_loss_tensor.item()

                            # eval metric
                            eval_precision_score, eval_recall_score, eval_f1_score, eval_iou_score = occ_eval_metric(cfg.eval_metric_list,output_logit,pc_query_label,cfg.occ_threshold)
                            running_eval_precision += (eval_precision_score.mean()).item()
                            running_eval_recall += (eval_recall_score.mean()).item()
                            running_eval_f1 += (eval_f1_score.mean()).item()
                            running_eval_iou += (eval_iou_score.mean()).item()  

                            eval_log_step += 1
                        
                        if rank == 0:
                            avg_eval_loss = running_eval_loss / eval_log_step
                            avg_eval_loss_occ = running_eval_loss_occ / eval_log_step
                            avg_eval_loss_kl = running_eval_loss_kl / eval_log_step
                            avg_eval_occ_precision = running_eval_precision / eval_log_step
                            avg_eval_occ_recall = running_eval_recall / eval_log_step
                            avg_eval_occ_f1 = running_eval_f1 / eval_log_step
                            avg_eval_occ_iou = running_eval_iou / eval_log_step

                            logger.info(f"\n eval_loss: {avg_eval_loss}, avg_eval_loss_occ: {avg_eval_loss_occ}, avg_eval_loss_kl: {avg_eval_loss_kl}, avg_eval_occ_precision: {avg_eval_occ_precision}, avg_eval_occ_recall: {avg_eval_occ_recall}, avg_eval_occ_f1: {avg_eval_occ_f1}, avg_eval_occ_iou: {avg_eval_occ_iou}, step: {step}, global_step: {global_step}, lr: {optimizer.param_groups[0]['lr']} \n")

                            writer.add_scalar("eval_loss", eval_loss.item(), global_step)
                            writer.add_scalar("eval_loss_occ", eval_loss_occ.item(), global_step)
                            if eval_loss_kl is not None:
                                writer.add_scalar("eval_loss_kl", eval_loss_kl.item(), global_step)
                            writer.add_scalar("eval_precision_score", eval_precision_score.mean().item(), global_step)
                            writer.add_scalar("eval_recall_score", eval_recall_score.mean().item(), global_step)
                            writer.add_scalar("eval_f1_score", eval_f1_score.mean().item(), global_step)
                            writer.add_scalar("eval_iou_score", eval_iou_score.mean().item(), global_step)
                            
                            inout_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step}", "input_output_results/val")
                            save_inout_pc(inout_dir, input_dict_val, output_logit)

                    val_dataloader.sampler.set_start_index(0)
                    start_step_eval = 0                    
                input_dict = next_batch
                # torch.npu.empty_cache()   
        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        # torch.npu.empty_cache()  
        train_dataloader.sampler.set_start_index(0)
        start_step = 0
        

if __name__ == "__main__":
    main()