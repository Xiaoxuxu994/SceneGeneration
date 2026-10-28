import argparse
import datetime
import itertools
import os
import random
import traceback
from accelerate import Accelerator
import hydra
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from copy import deepcopy
from easydict import EasyDict
import time
import json
import math
import sys
from PIL import Image
import shutil
from utils.basic import seed_anything, count_parameters

from datasets import create_dataloader
# from model.network import Network
from utils.misc import get_logger, is_logging_process, pretty_print_hydra_config, move_to_device, get_rank
from utils.basic import seed_anything
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.dist import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    setup_for_distributed,
)
from accelerate import DistributedDataParallelKwargs
from transformers.trainer_pt_utils import get_model_param_count
from accelerate import (
    DistributedType,
)
from accelerate.utils import (
    DataLoaderConfiguration,
    DynamoBackend,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    TorchDynamoPlugin,
    set_seed,
)
import numpy as np
from collections import defaultdict

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        with open_dict(cfg):
            cfg.job_logging_cfg = HydraConfig.get().job_logging

        # random seed
        if cfg.random_seed is None:
            cfg.random_seed = random.randint(1, 10000)
        seed_anything(cfg.random_seed, deterministic=False)              # deterministic=True for reproduction

        ## 1. Build accelerator
        self.build_accelerator()

        if is_logging_process():
            pretty_print_hydra_config(cfg)

        ## 2. Prepare model
        self.log_info("Preparing model...")
        self.model = self.prepare_model()
        self.n_learnable_parameters = get_model_param_count(
            self.model, trainable_only=True
        )
        self.n_fix_parameters = get_model_param_count(
            self.model, trainable_only=False
        )
        self.accelerator.wait_for_everyone()

        ## 3. Prepare dataloader
        self.log_info("Making train dataloader...")
        self.train_loader = create_dataloader(cfg, 'train')
        self.log_info("Making test dataloader...")
        self.test_loader = create_dataloader(cfg, 'test')
        self.accelerator.wait_for_everyone()

        ## 5. Prepare optimizer and scheduler (fsdp should after preparing the model using accelerate)
        if self.cfg.get("fsdp_plugin"):
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")
        else:
            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")

            # self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

        # Create the LR scheduler
        self.iters_per_epoch = self.cfg.train.iters_per_epoch if self.cfg.train.iters_per_epoch > 0 else len(self.train_loader)
        self.iters_per_test = self.cfg.test.iters_per_test if self.cfg.test.iters_per_test > 0 else len(self.test_loader)
        self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
        self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        self.lr_scheduler = build_scheduler(
            self.cfg.train.lr_scheduler, optimizer=self.optimizer
        )
        self.log_info(f"LRScheduler: {self.lr_scheduler}")

        ## 6. Prepare accelerate training
        self.prepare_training()

    def build_optimizer(self, cfg_optimizer, model, param_group_fn=None):
        return build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def prepare_training(self):
        # report model details
        self.log_info(
            f"total number of learnable params: {self.n_learnable_parameters / 1e6} M"
        )
        self.log_info(
            f"total number of fixed params: {self.n_fix_parameters / 1e6} M"
        )

        # Wrap the model, optmizer, and scheduler with accelerate
        self.log_info("before accelerator.prepare")
        
        ## ------------------------------------------------------------------------------------
        # # warp for enable deepspeed zero2
        # (
        #     self.model,
        #     self.train_loader,
        #     self.test_loader,
        #     self.optimizer,
        #     self.lr_scheduler
        # ) = self.accelerator.prepare(
        #     self.model, self.train_loader, self.test_loader, self.optimizer, self.lr_scheduler
        # )

        (
            self.model,
            self.optimizer,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        # prepare 之后重新统计 dataloader 长度，用于日志和后续 resume 计算
        prepared_iters_per_epoch = (
            self.cfg.train.iters_per_epoch
            if self.cfg.train.iters_per_epoch > 0
            else len(self.train_loader)
        )

        prepared_iters_per_test = (
            self.cfg.test.iters_per_test
            if self.cfg.test.iters_per_test > 0
            else len(self.test_loader)
        )
        # 如果和 prepare 前不一致，给一个提示
        if prepared_iters_per_epoch != self.iters_per_epoch:
            self.log_info(
                f"[Warn] iters_per_epoch changed after accelerator.prepare: "
                f"{self.iters_per_epoch} -> {prepared_iters_per_epoch}. "
                f"For DeepSpeed + OneCycleLR, it is recommended to set "
                f"`cfg.train.iters_per_epoch` explicitly."
            )

        self.iters_per_epoch = prepared_iters_per_epoch
        self.iters_per_test = prepared_iters_per_test
        ## ------------------------------------------------------------------------------------

        # # don't wrap dataloader
        # (
        #     self.optimizer,
        #     self.lr_scheduler,
        # ) = self.accelerator.prepare(
        #     self.optimizer, self.lr_scheduler
        # )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(os.path.basename(self.cfg.log.output_dir))

        # Report the training info
        self.total_batch_size = (
            self.cfg.train.batch_size
            * self.accelerator.num_processes
            * self.cfg.train.gradient_accumulation_steps
        )
        self.log_info("***** Running training *****")
        self.log_info(f"LR = {self.cfg.train.optimizer.lr:.8f}")
        self.log_info(f"Weigth Decay = {self.cfg.train.optimizer.weight_decay:.8f}")
        self.log_info(f"Instantaneous batch size per device = {self.cfg.train.batch_size}")
        self.log_info(f"Total Batch size = {self.total_batch_size}")
        self.log_info(
            f"Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}"
        )
        self.log_info(f"Number of epochs = {self.cfg.train.num_epoch}")
        self.log_info(
            f"Number of training steps per epoch = {self.iters_per_epoch}"
        )
        self.log_info(
            f"Number of total training steps = {self.iters_per_epoch * self.cfg.train.num_epoch}"
        )
        # self.log_info(f"Number of training examples per epoch = {len(self.dataloader.dataset)}")
        self.log_info(
            f"Number of model parameters = {self.n_fix_parameters / 1e6:.2f}M"
        )
        self.log_info(
            f"Number of model trainable parameters = {self.n_learnable_parameters / 1e6:.2f}M"
        )

        # Auto resume the checkpoint
        latest_epoch = self.auto_resume()
        self.initial_global_step = self.iters_per_epoch * latest_epoch
        self.first_epoch = latest_epoch
        self.global_step = self.initial_global_step 

        os.makedirs(self.cfg.log.ckpt_dir, exist_ok=True)

    def prepare_model(self):
        model = hydra.utils.instantiate(self.cfg.model)
        count_parameters(model)
        return model
    
    def before_epoch(self, epoch):
        pass

    def train(self):
        # Start Train!
        start_time = time.time()
        self.accelerator.wait_for_everyone()

        # Initialize variable to track the best validation metric
        best_val_metric = float('inf')  # For metrics like loss; use -float('inf') for accuracy
        best_model_path = None

        max_checkpoints = self.cfg.log.max_checkpoints  # Maximum number of recent checkpoints to keep
        saved_checkpoints = []  # List to track saved checkpoint paths

        for epoch in range(self.first_epoch, self.cfg.train.num_epoch):
            # torch.cuda.reset_peak_memory_stats()
            torch_npu.npu.reset_peak_memory_stats()

            self.before_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)
            # train_stats = {}

            # Perform validation at the end of each epoch
            val_stats = self.validate(epoch)
            # if self.accelerator.is_main_process:
            #     val_stats = self.validate(epoch)
            # else:
            #     val_stats = {}
            self.accelerator.wait_for_everyone()

            best_model_path = os.path.join(self.cfg.log.ckpt_dir, "best_model")

            if self.accelerator.is_main_process:
                current_val_metric = val_stats.get("loss", float('inf'))
                should_save_best = current_val_metric < best_val_metric
                if should_save_best:
                    best_val_metric = current_val_metric
            else:
                should_save_best = False

            # 广播保存决策
            flag = torch.tensor([int(should_save_best)], device=self.accelerator.device)
            dist.broadcast(flag, src=0)
            should_save_best = bool(flag.item())

            if should_save_best:
                self.accelerator.wait_for_everyone()
                self.accelerator.save_state(best_model_path, safe_serialization=False)
                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    self.log_info(f"Saved best model at epoch {epoch} with val_metric: {best_val_metric:.4f}")

            self.accelerator.wait_for_everyone()

            if ((epoch + 1) % self.cfg.log.ckpt_interval == 0) or (epoch + 1 == self.cfg.train.num_epoch):
                if self.accelerator.sync_gradients:
                    self.global_step = self.iters_per_epoch * (epoch + 1)
                    save_path = os.path.join(
                        self.cfg.log.ckpt_dir,
                        f"checkpoint_{epoch}",
                    )

                    self.accelerator.wait_for_everyone()
                    self.accelerator.save_state(save_path, safe_serialization=False)
                    self.accelerator.wait_for_everyone()

                    if self.accelerator.is_main_process:
                        self.log_info(f"Saved state for global step {self.global_step}")

                        saved_checkpoints.append(save_path)
                        if len(saved_checkpoints) > max_checkpoints:
                            oldest_checkpoint = saved_checkpoints.pop(0)
                            if os.path.exists(oldest_checkpoint):
                                shutil.rmtree(oldest_checkpoint)
                                self.log_info(f"Removed old checkpoint: {oldest_checkpoint}")


            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_learnable_parameters,
            }

            if self.accelerator.is_main_process:
                with open(
                    os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                self.log_all(log_stats, step=self.global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_info("Training time {}".format(total_time_str))

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def validate(self, epoch):
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Validation Epoch: [{epoch}]"

        val_loss = 0.0
        total_samples = 0.0

        # per-dataset weighted stats
        per_dataset_sums = defaultdict(lambda: defaultdict(float))
        per_dataset_counts = defaultdict(float)

        self.log_info(f"Start validation for epoch {epoch}")

        with torch.no_grad():
            for it, batch in enumerate(
                metric_logger.log_every(self.test_loader, self.cfg.test.print_freq, header)
            ):
                if it >= self.iters_per_test:
                    break

                if batch is None:
                    continue

                dataset_name = self._get_dataset_name(batch)
                batch_weight = len(batch)  # keep original semantics

                batch = move_to_device(batch, self.accelerator.device)

                # Forward pass
                fwd_outputs = self.forward_batch(batch, mode='test')
                outputs = self.calculate_loss(fwd_outputs, batch, mode='test') 
                metric_dict = self.calculate_metrics(fwd_outputs, batch, mode='test')
                loss = outputs.loss

                loss_value = loss.item()
                val_loss += loss_value * batch_weight
                total_samples += batch_weight

                # only keep scalar items for metric logger
                scalar_outputs = {}
                for item in outputs:
                    v = outputs[item]

                    if torch.is_tensor(v):
                        if v.ndim == 0:
                            v = v.item()
                        else:
                            continue
                    elif np.isscalar(v):
                        v = float(v)
                    else:
                        continue

                    scalar_outputs[item] = v

                # extra metrics
                for item, v in metric_dict.items():
                    if torch.is_tensor(v):
                        if v.ndim == 0:
                            v = v.item()
                        else:
                            continue
                    elif np.isscalar(v):
                        v = float(v)
                    else:
                        continue

                    scalar_outputs[item] = v

                # per-dataset aggregation: loss + metrics一起聚合
                if dataset_name is not None:
                    for k, v in scalar_outputs.items():
                        per_dataset_sums[dataset_name][k] += v * batch_weight
                    per_dataset_counts[dataset_name] += batch_weight

                metric_logger.update(**scalar_outputs)

        # ---------------------------
        # synchronize metric_logger meters across all processes
        # ---------------------------
        metric_logger.synchronize_between_processes()

        # ---------------------------
        # globally reduce total val loss / total samples
        # ---------------------------
        loss_cnt = torch.tensor(
            [val_loss, total_samples],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        dist.all_reduce(loss_cnt, op=dist.ReduceOp.SUM)

        global_val_loss = loss_cnt[0].item()
        global_total_samples = loss_cnt[1].item()

        if global_total_samples > 0:
            global_val_loss /= global_total_samples

        if self.accelerator.is_main_process:
            self.log_info(f"Validation results: {metric_logger}")

        # ---------------------------
        # reduce per-dataset stats across all processes
        # ---------------------------
        ds_log = {}

        # gather all dataset names from every rank
        local_dataset_names = list(per_dataset_counts.keys())
        gathered_dataset_names = [None for _ in range(self.accelerator.num_processes)]
        dist.all_gather_object(gathered_dataset_names, local_dataset_names)

        all_dataset_names = sorted(
            set(name for names in gathered_dataset_names for name in (names or []))
        )

        for ds_name in all_dataset_names:
            # reduce count
            cnt_local = per_dataset_counts.get(ds_name, 0.0)
            cnt_tensor = torch.tensor(
                [cnt_local], device=self.accelerator.device, dtype=torch.float32
            )
            dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
            cnt_global = cnt_tensor.item()

            if cnt_global <= 0:
                continue

            # gather all metric keys for this dataset
            local_metric_keys = list(per_dataset_sums.get(ds_name, {}).keys())
            gathered_metric_keys = [None for _ in range(self.accelerator.num_processes)]
            dist.all_gather_object(gathered_metric_keys, local_metric_keys)

            all_metric_keys = sorted(
                set(k for keys in gathered_metric_keys for k in (keys or []))
            )

            for k in all_metric_keys:
                s_local = per_dataset_sums.get(ds_name, {}).get(k, 0.0)
                s_tensor = torch.tensor(
                    [s_local], device=self.accelerator.device, dtype=torch.float32
                )
                dist.all_reduce(s_tensor, op=dist.ReduceOp.SUM)
                s_global = s_tensor.item()

                ds_log[f"val_dataset/{ds_name}/{k}"] = s_global / cnt_global

        if self.accelerator.is_main_process and len(ds_log) > 0:
            self.accelerator.log(ds_log, step=self.global_step)

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats["loss"] = global_val_loss   # use exact globally reduced validation loss
        return stats

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        loss_details_dict = {}

        per_dataset_sums = defaultdict(lambda: defaultdict(float))
        per_dataset_counts = defaultdict(int)

        start_steps = epoch * self.iters_per_epoch
        self.global_step = start_steps

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        # 慢数据定位：每 straggler_log_freq 步汇总各卡这段时间里"等数据"最久的一次和对应数据集（0 = 关闭）
        strag_freq = int(self.cfg.train.get("straggler_log_freq", 0) or 0)
        strag_win, t_prev = [], time.time()

        for it, batch in enumerate(metric_logger.log_every(
            self.train_loader, self.cfg.train.print_freq, header
        )):
            if it >= self.iters_per_epoch:
                break

            dataset_name = self._get_dataset_name(batch)
            if strag_freq > 0:
                strag_win.append((time.time() - t_prev, dataset_name))

            with self.accelerator.accumulate(self.model):
                # move to device
                batch = move_to_device(batch, device=self.accelerator.device)

                # forward
                with self.accelerator.autocast():
                    forward_output = self.forward_batch(batch, mode='train')

                # loss
                batch_output = self.calculate_loss(forward_output, batch, mode='train')
                loss = batch_output.loss
                if loss > self.cfg.train.clip_loss:
                    loss = loss * 0.0

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    self.log_info(
                        f"NaN/Inf loss at epoch={epoch}, iter={it}, global_step={self.global_step}, "
                        f"dataset={dataset_name}, loss={loss_value}"
                    )
                    sys.exit(1)

                # backward
                self.accelerator.backward(loss)

                scalar_loss_items = {}
                for item in batch_output:
                    if 'loss' in item:
                        batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                        scalar_loss_items[item] = batch_output[item]

                        if item in loss_details_dict:
                            loss_details_dict[item] += (
                                batch_output[item] / self.cfg.train.gradient_accumulation_steps
                                if loss_value != 0 else 0.0
                            )
                        else:
                            loss_details_dict[item] = (
                                batch_output[item] / self.cfg.train.gradient_accumulation_steps
                                if loss_value != 0 else 0.0
                            )

                if dataset_name is not None and loss_value != 0:
                    per_dataset_counts[dataset_name] += 1
                    for k, v in scalar_loss_items.items():
                        per_dataset_sums[dataset_name][k] += v

                # grad clip
                if self.accelerator.sync_gradients:
                    params_to_clip = self.model.parameters()
                    self.accelerator.clip_grad_norm_(params_to_clip, self.cfg.train.clip_grad)

                    def get_gradient_norm(parameters):
                        norm = 0.0
                        for param in parameters:
                            if param.grad is None:
                                continue
                            local_norm = param.grad.detach().data.norm(2)
                            norm += local_norm.item() ** 2
                        return norm ** 0.5

                    # DeepSpeed（ZeRO-2）下梯度在 DeepSpeed 自己的分片缓冲里，param.grad 是 None，
                    # 逐参数求范数恒为 0；accelerate 的 backward 已经调用了 engine.step()，直接取它算好的全局梯度范数
                    ds_norm = (self.model.get_global_grad_norm()
                               if hasattr(self.model, "get_global_grad_norm") else None)
                    if ds_norm is not None:
                        grad_norm = float(ds_norm)
                    else:
                        grad_norm = get_gradient_norm(self.model.parameters())
                else:
                    grad_norm = 0.0

                # optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            if self.accelerator.sync_gradients:
                start_steps += 1

                batch_output.update(loss_details_dict)
                loss_details_dict = {}

                if start_steps % 10 == 0:
                    self.log_all(batch_output, start_steps, prefix='train')
                metric_logger.update(**batch_output)

                min_lr = 10.0
                max_lr = 0.0
                for group in self.optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                self.accelerator.log({"lr": max_lr}, step=start_steps)
                self.accelerator.log({"min_lr": min_lr}, step=start_steps)

                weight_decay_value = None
                for group in self.optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)
                metric_logger.update(grad_norm=grad_norm)
                self.accelerator.log({"weight_decay": weight_decay_value}, step=start_steps)
                self.accelerator.log({"grad_norm": grad_norm}, step=start_steps)

                if self.accelerator.is_main_process:
                    ds_log = {}
                    for ds_name, cnt in per_dataset_counts.items():
                        if cnt <= 0:
                            continue
                        for k, s in per_dataset_sums[ds_name].items():
                            ds_log[f"train_dataset/{ds_name}/{k}"] = s / cnt
                    if len(ds_log) > 0:
                        self.accelerator.log(ds_log, step=start_steps)

                self.global_step = start_steps

            if strag_freq > 0:
                if (it + 1) % strag_freq == 0:
                    self._log_stragglers(strag_win, epoch, it)
                    strag_win = []
                t_prev = time.time()

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def _log_stragglers(self, window, epoch, it):
        """各卡把本窗口内等数据最久的一次（耗时、数据集）和平均等待发给 rank 0，rank 0 打印最慢的几张卡
        与按数据集汇总的最长等待。等数据的卡会让其余所有卡在梯度同步时陪等，所以整步耗时由这里最慢的卡决定。"""
        if not window:
            return
        worst = max(window, key=lambda x: x[0])
        local = dict(rank=self.accelerator.process_index, worst_s=float(worst[0]), worst_ds=str(worst[1]),
                     mean_s=float(sum(x[0] for x in window) / len(window)),
                     per_ds={})
        for t, d in window:
            d = str(d)
            a = local["per_ds"].setdefault(d, [0, 0.0, 0.0])
            a[0] += 1; a[1] += t; a[2] = max(a[2], t)
        if dist.is_available() and dist.is_initialized():
            allv = [None] * dist.get_world_size()
            dist.all_gather_object(allv, local)
        else:
            allv = [local]
        if not self.accelerator.is_main_process:
            return
        allv.sort(key=lambda x: -x["worst_s"])
        top = "  ".join(f"rank{x['rank']}:{x['worst_s']:.1f}s({x['worst_ds']})" for x in allv[:5])
        means = [x["mean_s"] for x in allv]
        self.log_info(f"[straggler] epoch {epoch} it {it + 1 - len(window)}-{it}: slowest data waits {top} | "
                      f"mean wait per rank min {min(means):.2f}s max {max(means):.2f}s")
        agg = {}
        for x in allv:
            for d, (n, tot, mx) in x["per_ds"].items():
                b = agg.setdefault(d, [0, 0.0, 0.0])
                b[0] += n; b[1] += tot; b[2] = max(b[2], mx)
        rows = sorted(agg.items(), key=lambda kv: -kv[1][2])[:8]
        self.log_info("[straggler]   by dataset (n / mean wait / max wait): " +
                      "  ".join(f"{d} {n}/{tot / n:.1f}s/{mx:.1f}s" for d, (n, tot, mx) in rows))

    def log_all(self, output, step, prefix=""):        
        if 'log_keys' in output:
            log_keys = output.log_keys
        else:
            log_keys = list(output.keys())

        tb_alias_map = {
            "train_loss": ["loss/total"],
            "train_local_pts_loss": ["loss/point_local"],
            "train_trans_loss": ["loss/extr_t"],
            "train_rot_loss": ["loss/extr_r"],
            "train_normal_loss": ["loss/normalize"],
            "val_loss": ["loss_val/total"],
            "val_local_pts_loss": ["loss_val/point_local"],
            "val_trans_loss": ["loss_val/extr_t"],
            "val_rot_loss": ["loss_val/extr_r"],
            "val_normal_loss": ["loss_val/normalize"],
        }

        log_scaler = {}
        log_img = {}
        for k in log_keys:
            v = output[k]
            if np.isscalar(v):
                log_scaler[prefix+'/'+k] = v
                # 额外 alias 名字
                if k in tb_alias_map:
                    for alias_k in tb_alias_map[k]:
                        alias_key = f"{prefix}/{alias_k}" if prefix else alias_k
                        log_scaler[alias_key] = v

                continue
            
            if isinstance(v, Image.Image):
                log_img[prefix+'/'+k] = v

        self.accelerator.log(log_scaler, step)
        for tracker in self.accelerator.trackers:
            tracker.log_images(log_img, step)

    def _get_dataset_name(self, batch):
        """
        batch is expected to be a list of dicts.
        Each sample dict contains a 'dataset' field.
        """
        if not isinstance(batch, (list, tuple)) or len(batch) == 0:
            return None

        names = []
        for sample in batch:
            if not isinstance(sample, dict):
                continue
            if "dataset" not in sample:
                continue

            v = sample["dataset"]
            if isinstance(v, str):
                names.append(v)
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                names.extend([str(x) for x in v])
            else:
                names.append(str(v))

        if len(names) == 0:
            return None

        uniq = sorted(set(names))
        if len(uniq) == 1:
            return uniq[0]
        return "MIXED_" + "_".join(uniq)

    def _normalize_dataset_name(self, v):
        """
        Normalize dataset name into a plain string.
        """
        if v is None:
            return None

        if isinstance(v, str):
            return v

        if isinstance(v, (list, tuple)) and len(v) > 0:
            uniq = sorted(set(str(x) for x in v))
            if len(uniq) == 1:
                return uniq[0]
            return "MIXED_" + "_".join(uniq)

        return str(v)
    
    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    def calculate_loss(self, output, batch, mode='train'):
        pass

    def calculate_metrics(self, output, batch, mode='test'):
        return {}

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later

        if self.cfg.log.use_wandb:
            log_with = 'wandb'
        elif self.cfg.log.use_tensorboard:
            log_with = 'tensorboard'
        else:
            log_with = 'all'

        mixed_precision = 'no' if self.cfg.train.model_dtype not in ['fp8', 'fp16', 'bf16'] else self.cfg.train.model_dtype

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # dynamic complie
        if self.cfg.train.get("dynamo_backend"):
            if isinstance(self.cfg.train.dynamo_backend, str) and hasattr(
                DynamoBackend, self.cfg.train.dynamo_backend.upper()
            ):
                dynamo_backend = getattr(DynamoBackend, self.cfg.train.dynamo_backend.upper())
            elif isinstance(self.cfg.train.dynamo_backend, DynamoBackend):
                dynamo_backend = self.cfg.train.dynamo_backend
            else:
                print(
                    f"Invalid dynamo_backend {self.cfg.train.dynamo_backend}, using default. Please refer to "
                    "https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.DynamoBackend for available names."
                )
        else:
            dynamo_backend = DynamoBackend.NO

        print(f"Using dynamo backend: {dynamo_backend}")

        torch._inductor.config.reorder_for_compute_comm_overlap = True

        # dynamo_plugin = TorchDynamoPlugin(
        #     backend=dynamo_backend,
        #     mode="max-autotune-no-cudagraphs",
        #     dynamic=self.cfg.train.get("dynamic_compile", True),
        # )
        
        accelerate_config = dict(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=True,
                split_batches=False,
                dispatch_batches=None,
                even_batches=True, # default is true
                use_seedable_sampler=False,
            ),
            step_scheduler_with_optimizer=False,             # not to step n_gpus times per step.
            # dynamo_plugin=dynamo_plugin,
            dynamo_backend=dynamo_backend,
        )

        # fsdp
        if self.cfg.get("fsdp_plugin"):
            fsdp_plugin_kwargs = {}
            fsdp_plugin_kwargs[
                "mixed_precision_policy"
            ] = torch.distributed.fsdp.MixedPrecision(
                param_dtype=self.weight_dtype,
                reduce_dtype=self.weight_dtype,
                buffer_dtype=self.weight_dtype,
                cast_forward_inputs=True,
                cast_root_forward_inputs=True,
            )

            fsdp_plugin = hydra.utils.instantiate(self.cfg.fsdp_plugin)(**fsdp_plugin_kwargs)
            accelerate_config["fsdp_plugin"] = fsdp_plugin
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.cfg.train.find_unused_parameters)
            accelerate_config['kwargs_handlers'] = [ddp_kwargs]

        accelerator = Accelerator(**accelerate_config)

        self.logger = get_logger(self.cfg, os.path.basename(__file__))

        # To block the print on non main process
        setup_for_distributed(accelerator.is_main_process)

        self.log_info(accelerator.state)

        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed, device_specific=True)

        self.device = accelerator.device

        self.accelerator = accelerator

    def auto_resume(self):
        if self.cfg.train.resume:
            path = self.cfg.train.resume
        elif os.path.exists(self.cfg.log.ckpt_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.log.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint_")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is not None:
                path = os.path.join(self.cfg.log.ckpt_dir, path)
        else:
            path = None

        if path is None:
            self.log_info("Checkpoint does not exist. Starting a new training run.")
            
            start_epoch = 0
        else:
            self.log_info(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(
                path,
                load_module_strict=False,
                # load_module_only=True,
            )
            # Extract epoch number from checkpoint path
            # Handles both "checkpoint_N" and "best_model" formats
            if "checkpoint_" in path:
                # Extract epoch from "checkpoint_N" format
                checkpoint_name = path.rstrip('/').split('/')[-1]
                ckpt_epoch = int(checkpoint_name.split("checkpoint_")[-1])
                start_epoch = ckpt_epoch + 1
                self.log_info(f"Resume from checkpoint epoch {ckpt_epoch}, start from epoch {start_epoch}")
            else:
                # For "best_model" or other formats, start from epoch 0
                # This is correct for stage transitions where we want to reset the epoch counter
                start_epoch = 0

        return start_epoch

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
