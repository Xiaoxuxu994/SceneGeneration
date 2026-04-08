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

        # ### -------------------------------------------------------------------------
        # name_map = {id(p): n for n, p in self.model.named_parameters()}
        # for i, group in enumerate(self.optimizer.param_groups):
        #     counter = {
        #         "encoder": 0,
        #         "decoder": 0,
        #         "point_decoder": 0,
        #         "camera_decoder": 0,
        #         "point_head": 0,
        #         "camera_head": 0,
        #         "others": 0,
        #     }
        #     for p in group["params"]:
        #         n = name_map.get(id(p), "")
        #         if n.startswith("encoder."):
        #             counter["encoder"] += 1
        #         elif n.startswith("decoder."):
        #             counter["decoder"] += 1
        #         elif n.startswith("point_decoder."):
        #             counter["point_decoder"] += 1
        #         elif n.startswith("camera_decoder."):
        #             counter["camera_decoder"] += 1
        #         elif n.startswith("point_head."):
        #             counter["point_head"] += 1
        #         elif n.startswith("camera_head."):
        #             counter["camera_head"] += 1
        #         else:
        #             counter["others"] += 1
        #     print(f"\n===== group {i} =====")
        #     print("lr =", group["lr"], "wd =", group["weight_decay"])
        #     print(counter)
        # ### -------------------------------------------------------------------------

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
        # warp for enable deepspeed zero2
        (
            self.model,
            self.train_loader,
            self.test_loader,
            self.optimizer,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model, self.train_loader, self.test_loader, self.optimizer, self.lr_scheduler
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
            if self.accelerator.is_main_process:
                val_stats = self.validate(epoch)
            else:
                val_stats = {}
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
        total_samples = 0

        self.log_info(f"Start validation for epoch {epoch}")
        with torch.no_grad():
            for batch in metric_logger.log_every(
                self.test_loader, self.cfg.train.print_freq, header
            ):
                batch = move_to_device(batch, self.accelerator.device)

                # Forward pass
                outputs = self.forward_batch(batch, mode='test')
                outputs = self.calculate_loss(outputs, batch, mode='train')
                loss = outputs.loss

                # Gather statistics
                loss_value = loss.item()
                val_loss += loss_value * len(batch)
                total_samples += len(batch)

                # self.log_all(outputs, self.global_step, prefix='val')

                metric_logger.update(**outputs)

        # Average the validation loss
        val_loss /= total_samples

        # Gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        self.log_info(f"Validation results: {metric_logger}")

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        # metric_logger.add_meter(
        #     "dataloader", SmoothedValue(window_size=1, fmt="{value:.6f}")
        # )
        header = "Epoch: [{}]".format(epoch)
        loss_details_dict = {}
        start_steps = epoch * self.iters_per_epoch
        self.global_step = start_steps

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        for it, batch in enumerate(metric_logger.log_every(
            self.train_loader, self.cfg.train.print_freq, header
        )):
            if it >= self.iters_per_epoch:
                break

            with self.accelerator.accumulate(self.model):
                # Perform the forward using the accerlate
                batch = move_to_device(batch, device=self.accelerator.device)
                with self.accelerator.autocast():
                    forward_output = self.forward_batch(batch, mode='train')
                batch_output = self.calculate_loss(forward_output, batch, mode='train')
                loss = batch_output.loss
                if loss > self.cfg.train.clip_loss:
                    loss = loss * 0.0

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    rank = get_rank()
                    print(
                        f"Rank {rank}: Loss is {loss_value}, stopping training at iter {it} (epoch {epoch}, global step {self.global_step}).",
                        force=True,
                    )
                    sys.exit(1)

                self.accelerator.backward(loss)
                # ### debug
                # model_for_check = self.model.module if hasattr(self.model, "module") else self.model

                # no_grad_names = []
                # for name, p in model_for_check.named_parameters():
                #     if p.requires_grad and p.grad is None:
                #         no_grad_names.append(name)

                # print("\n[NO GRAD PARAMS]")
                # for n in no_grad_names[:300]:
                #     print(n)
                # print("total:", len(no_grad_names))
                # ### debug
                
                for item in batch_output:
                    if 'loss' in item:
                        batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                        if item in loss_details_dict:
                            loss_details_dict[item] += batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                        else:
                            loss_details_dict[item] = batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0

                # clip the gradient
                if self.accelerator.sync_gradients:
                    params_to_clip = self.model.parameters()
                    self.accelerator.clip_grad_norm_(
                        params_to_clip, self.cfg.train.clip_grad
                    )

                    def get_gradient_norm(parameters):
                        norm = 0
                        for param in parameters:
                            if param.grad is None:
                                continue
                            local_norm = param.grad.detach().data.norm(2)
                            norm += local_norm.item() ** 2
                        norm = norm**0.5
                        return norm

                    grad_norm = get_gradient_norm(self.model.parameters())

                # if self.accelerator.state.deepspeed_plugin is None:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                # self.lr_scheduler.step()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            if self.accelerator.sync_gradients:
                    start_steps += 1

                    # Report to tensorboard
                    batch_output.update(loss_details_dict)
                    loss_details_dict = {}

                    if start_steps % 10 == 0 :
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
                    self.accelerator.log(
                        {"weight_decay": weight_decay_value}, step=start_steps
                    )
                    self.accelerator.log({"grad_norm": grad_norm}, step=start_steps)

                    self.global_step = start_steps

        # # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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

    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    def calculate_loss(self, output, batch, mode='train'):
        pass

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        # Initialize the Environment variables throught MPI run
        # import os
        # print(f"[before init_distributed_mode] WORLD_SIZE={os.environ.get('WORLD_SIZE')}, RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later
        # print(f"[after init_distributed_mode] WORLD_SIZE={os.environ.get('WORLD_SIZE')}, RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")

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
                even_batches=False, #default is True
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

        # self.logger.rank_zero_only = False
        self.log_info(accelerator.state)
        # self.logger.rank_zero_only = True

        # ### debug------------------------------------------------
        # import os
        # import torch.distributed as dist

        # print(f"WORLD_SIZE env: {os.environ.get('WORLD_SIZE')}")
        # print(f"RANK env: {os.environ.get('RANK')}")
        # print(f"LOCAL_RANK env: {os.environ.get('LOCAL_RANK')}")
        # print(f"accelerator.num_processes: {accelerator.num_processes}")
        # print(f"accelerator.process_index: {accelerator.process_index}")
        # print(f"accelerator.local_process_index: {accelerator.local_process_index}")

        # if dist.is_available() and dist.is_initialized():
        #     print(f"dist world_size: {dist.get_world_size()}")
        #     print(f"dist rank: {dist.get_rank()}")
        # else:
        #     print("torch.distributed not initialized")
        # ### debug------------------------------------------------


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
                # os.path.join(self.cfg.log.ckpt_dir, path)
                path
            )
            # Extract epoch number from checkpoint path
            # Handles both "checkpoint_N" and "best_model" formats
            if "checkpoint_" in path:
                # Extract epoch from "checkpoint_N" format
                checkpoint_name = path.rstrip('/').split('/')[-1]
                start_epoch = int(checkpoint_name.split("checkpoint_")[-1])
            else:
                # For "best_model" or other formats, start from epoch 0
                # This is correct for stage transitions where we want to reset the epoch counter
                start_epoch = 0

        return start_epoch

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
