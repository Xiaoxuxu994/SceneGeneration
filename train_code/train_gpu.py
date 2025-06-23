# %%
import os

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.adamw import AdamW
import numpy as np
import random

## colossalai
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

## opensora 
from opensora_local.acceleration.checkpoint import set_grad_checkpoint
from opensora_local.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora_local.acceleration.plugin import ZeroSeqParallelPlugin
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
from src.dataset.dataset_build_scannet_v2 import DatasetForScanNetV2
from src.utils.misc import save_inout_pc,occ_eval_metric

import gc

print(os.environ['MASTER_PORT'])

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect() 

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    print("====== cfg: ",cfg)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print("====== exp_dir: ", exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp32", "fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    device = get_current_device()
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
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
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
    booster = Booster(plugin=plugin)
    
    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    # DatasetForScanNet
    train_dataset = DatasetForScanNetV2(
        data_root=cfg.data_root,
        data_list_name=cfg.train_data_list_name,
        query_sample_rate=cfg.query_sample_rate,
        max_query_number=cfg.max_query_number,
        Gridsample_mode = 'train',
        grid_size = cfg.grid_size
    )

    val_dataset = DatasetForScanNetV2(
        data_root=cfg.data_root,
        data_list_name=cfg.val_data_list_name,
        query_sample_rate=cfg.query_sample_rate,
        max_query_number=cfg.max_query_number,
        Gridsample_mode = 'train',
        grid_size = cfg.grid_size
    )

    train_dataloader = prepare_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
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

    logger.info(f"Train Dataset contains {len(train_dataset)} data: ({cfg.train_data_list_name})")
    logger.info(f"Val Dataset contains {len(val_dataset)} data: ({cfg.val_data_list_name})")

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
    # optimizer = HybridAdam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay, adamw_mode=True
    # )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # lr_scheduler = None
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cos_total_iter, eta_min=cfg.cos_eta_min)

    # 4.5. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)

    # 4.6. set training mode
    model.train()

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=train_dataloader
    )
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
        start_epoch, start_step, sampler_start_idx = load(booster, model, optimizer, lr_scheduler, cfg.resume)
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
    running_loss ,running_loss_occ = 0.0, 0.0
    loss_occ = torch.tensor(0).to(device, dtype)

    # bceloss
    occ_criterion = torch.nn.BCEWithLogitsLoss()

    # 8.2. Eval Metric related   
    running_iou, running_precision, running_recall, running_f1 = 0.0,0.0,0.0,0.0


    # =======================================================
    # 9. Training Loop
    # =======================================================
    for epoch in range(start_epoch, cfg.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        traindataloader_iter = iter(train_dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                input_dict = next(traindataloader_iter)
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key][0].to(device)

                model.train()
                outputs = model.forward(input_dict)
                # if dist.get_rank() == 0:
                #     print(outputs, input_dict)
                # exit(-1)

                pc_query_label = input_dict['pc_query_label'].unsqueeze(0)
                output_logit = outputs['logits']

                loss = torch.tensor(0).to(device, dtype)

                loss_occ = cfg.loss_weight_occ * occ_criterion(output_logit, pc_query_label)
                running_loss_occ += loss_occ.item()
                if (not torch.isnan(loss_occ)) and (not torch.isinf(loss_occ)):
                    loss += loss_occ                      
                
                ## update
                optimizer.zero_grad()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()

                # update scheduler after each iter
                if lr_scheduler is not None:
                    lr_scheduler.step()   

                ## cal eval
                precision_score, recall_score, f1_score, iou_score = occ_eval_metric(cfg.eval_metric_list,output_logit,pc_query_label,cfg.occ_threshold)

                running_precision += (precision_score.mean()).item()
                running_recall += (recall_score.mean()).item()
                running_f1 += (f1_score.mean()).item()
                running_iou += (iou_score.mean()).item()  

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()

                global_step = epoch * num_steps_per_epoch + step
                log_step += 1

                # Log to tensorboard
                if coordinator.is_master() and (global_step) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    avg_loss_occ = running_loss_occ / log_step
                    avg_occ_precision = running_precision / log_step
                    avg_occ_recall = running_recall / log_step
                    avg_occ_f1 = running_f1 / log_step
                    avg_occ_iou = running_iou / log_step

                    logger.info(f"\n loss: {avg_loss}, avg_loss_occ: {avg_loss_occ}, avg_occ_precision: {avg_occ_precision}, avg_occ_recall: {avg_occ_recall}, avg_occ_f1: {avg_occ_f1}, avg_occ_iou: {avg_occ_iou}, step: {step}, global_step: {global_step}, lr: {optimizer.param_groups[0]['lr']} \n")
                    pbar.set_postfix({"loss": avg_loss,"avg_loss_occ": avg_loss_occ, "avg_occ_precision": avg_occ_precision,"avg_occ_recall": avg_occ_recall,"avg_occ_f1": avg_occ_f1,"avg_occ_iou": avg_occ_iou, "step": step, "global_step": global_step, "lr": optimizer.param_groups[0]['lr']})
                    pbar.set_postfix_str('')

                    running_loss,running_loss_occ = 0.0,0.0
                    running_iou, running_precision, running_recall, running_f1 = 0.0,0.0,0.0,0.0
                    log_step = 0

                    writer.add_scalar("loss", loss.item(), global_step)
                    writer.add_scalar("loss_occ", loss_occ.item(), global_step)
                    writer.add_scalar("train_precision_score", precision_score.mean().item(), global_step)
                    writer.add_scalar("train_recall_score", recall_score.mean().item(), global_step)
                    writer.add_scalar("train_f1_score", f1_score.mean().item(), global_step)
                    writer.add_scalar("train_iou_score", iou_score.mean().item(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)
                    
                    # del tensor
                    del loss, loss_occ
                    del precision_score, recall_score, f1_score, iou_score
                    torch.cuda.empty_cache()

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step) % cfg.ckpt_every == 0:
                    save(booster, model, optimizer, lr_scheduler, epoch, step, global_step, cfg.batch_size, coordinator, exp_dir)
                    logger.info(f'-------------------------------------------------------------------------------')
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step} global_step {global_step} to {exp_dir}"
                    )
                    logger.info(f'-------------------------------------------------------------------------------')
                    inout_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step}", "input_output_results/train")
                    if coordinator.is_master():
                        save_inout_pc(inout_dir, input_dict, output_logit)

                    ### Eval Set
                    if cfg.enable_occ_eval:
                        logger.info(f'-------------------------------------------------------------------------------')
                        logger.info(
                            f"Evaluating {epoch} step {step} global_step {global_step}......"
                        )
                        logger.info(f'-------------------------------------------------------------------------------')
                        eval_log_step = 0
                        running_eval_loss, running_eval_loss_occ = 0.0, 0.0
                        running_eval_precision, running_eval_recall, running_eval_f1, running_eval_iou = 0.0, 0.0, 0.0, 0.0
                        val_dataloader.sampler.set_epoch(epoch)
                        valdataloader_iter = iter(val_dataloader)
                        with tqdm(
                                range(start_step_eval, num_steps_per_epoch_val),
                                desc=f"Epoch {epoch}",
                                disable=not coordinator.is_master(),
                                total=num_steps_per_epoch_val,
                                initial=start_step_eval,
                            ) as pbar_val:
                            for _ in pbar_val:
                                input_dict = next(valdataloader_iter)
                                for key in input_dict.keys():
                                    if isinstance(input_dict[key], torch.Tensor):
                                        input_dict[key] = input_dict[key][0].to(device)

                                model.eval()
                                with torch.no_grad():
                                    outputs = model.forward(input_dict)

                                pc_query_label = input_dict['pc_query_label'].unsqueeze(0)
                                output_logit = outputs['logits']

                                eval_loss = torch.tensor(0).to(device, dtype)
                                # eval loss
                                eval_loss_occ = cfg.loss_weight_occ * occ_criterion(output_logit.detach(), pc_query_label.detach())
                                running_eval_loss_occ += eval_loss_occ.item()                 
                                if (not torch.isnan(eval_loss_occ)) and (not torch.isinf(eval_loss_occ)):
                                    eval_loss += eval_loss_occ                                 

                                # Log loss values:
                                all_reduce_mean(eval_loss)
                                running_eval_loss += eval_loss.item()

                                # eval metric
                                eval_precision_score, eval_recall_score, eval_f1_score, eval_iou_score = occ_eval_metric(cfg.eval_metric_list,output_logit,pc_query_label,cfg.occ_threshold)
                                running_eval_precision += (eval_precision_score.mean()).item()
                                running_eval_recall += (eval_recall_score.mean()).item()
                                running_eval_f1 += (eval_f1_score.mean()).item()
                                running_eval_iou += (eval_iou_score.mean()).item()  

                                eval_log_step += 1
                            
                            if coordinator.is_master():
                                avg_eval_loss = running_eval_loss / eval_log_step
                                avg_eval_loss_occ = running_eval_loss_occ / eval_log_step
                                avg_eval_occ_precision = running_eval_precision / eval_log_step
                                avg_eval_occ_recall = running_eval_recall / eval_log_step
                                avg_eval_occ_f1 = running_eval_f1 / eval_log_step
                                avg_eval_occ_iou = running_eval_iou / eval_log_step

                                logger.info(f"\n eval_loss: {avg_eval_loss}, avg_eval_loss_occ: {avg_eval_loss_occ}, avg_eval_occ_precision: {avg_eval_occ_precision}, avg_eval_occ_recall: {avg_eval_occ_recall}, avg_eval_occ_f1: {avg_eval_occ_f1}, avg_eval_occ_iou: {avg_eval_occ_iou}, step: {step}, global_step: {global_step}, lr: {optimizer.param_groups[0]['lr']} \n")

                                writer.add_scalar("eval_loss", eval_loss.item(), global_step)
                                writer.add_scalar("eval_loss_occ", eval_loss_occ.item(), global_step)
                                writer.add_scalar("eval_precision_score", eval_precision_score.mean().item(), global_step)
                                writer.add_scalar("eval_recall_score", eval_recall_score.mean().item(), global_step)
                                writer.add_scalar("eval_f1_score", eval_f1_score.mean().item(), global_step)
                                writer.add_scalar("eval_iou_score", eval_iou_score.mean().item(), global_step)
                                
                                # del tensor
                                del eval_loss, eval_loss_occ
                                del eval_precision_score, eval_recall_score, eval_f1_score, eval_iou_score
                                torch.cuda.empty_cache()

                                inout_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step}", "input_output_results/val")
                                save_inout_pc(inout_dir, input_dict, output_logit)

                        val_dataloader.sampler.set_start_index(0)
                        start_step_eval = 0

                        ## cleanup
                        if coordinator.is_master():
                            cleanup_memory()                        

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        train_dataloader.sampler.set_start_index(0)
        start_step = 0
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
    
    
    
