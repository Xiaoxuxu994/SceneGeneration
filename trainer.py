import accelerate
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
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
from torch.utils.data import DataLoader, default_collate

from src.utils.train_utils import _filter_dataset, _filter_dataset_new, saveRuntimeCode


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # 可以在 training_step 做判断
    return default_collate(batch)


def patch_deepspeed(accelerator: accelerate.Accelerator, model):
    print("========================debug(after prepare)====================")
    print("[acc]", accelerate.__version__,
        " dist:", accelerator.state.distributed_type,
        " ds_plugin?:", getattr(accelerator.state, "deepspeed_plugin", None) is not None,
        " acc.deepspeed_engine_wrapped?:", bool(getattr(accelerator, "deepspeed_engine_wrapped", False)),
        " ds_ver:", getattr(ds, "__version__", "NA"))

    engine = None
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    rank0 = accelerator.is_main_process

    # 1) 在 Accelerate 1.0+，prepare 后的 model 有可能就是 DeepSpeedEngine
    if accelerate.utils.is_deepspeed_available() \
        and accelerator.state.distributed_type.name == "DEEPSPEED":
        if isinstance(model, ds.runtime.engine.DeepSpeedEngine):
            engine = model
        # 2) 有些场景 plugin 持有 engine 引用
        elif plugin is not None and hasattr(plugin, "deepspeed_engine"):
            engine = plugin.deepspeed_engine

    if rank0: print("engine is DeepSpeedEngine?:", isinstance(engine, ds.runtime.engine.DeepSpeedEngine))

    # ---- 打印 Zero/grad-clip 配置（以 plugin 为准）----
    zero_stage = getattr(plugin, "zero_stage", None) if plugin is not None else None
    grad_clip_cfg = None
    if plugin is not None:
        # Accelerate 1.0.x: plugin.deepspeed_config 是 dict（或 HF DS Config 对象），尽量兼容取值
        cfg = getattr(plugin, "deepspeed_config", None)
        if isinstance(cfg, dict):
            grad_clip_cfg = cfg.get("gradient_clipping", None)
    if rank0: print(f"zero_stage: {zero_stage}, ds_config.gradient_clipping: {grad_clip_cfg}")
    if rank0: print("================================================================")

    try:
        import types
        # A) 配置层：关掉 plugin 里的 gradient_clipping
        if plugin is not None and isinstance(getattr(plugin, "deepspeed_config", None), dict):
            plugin.deepspeed_config["gradient_clipping"] = 0.0

        if isinstance(engine, ds.runtime.engine.DeepSpeedEngine):
            # B1) 引擎 config 是 dict 的情况（DS 0.17.x 常见）
            if isinstance(getattr(engine, "config", None), dict):
                engine.config["gradient_clipping"] = 0.0
                # 部分版本 optimizer 的 param_groups 里也有 max_grad_norm，兜底置 0
                if hasattr(engine, "optimizer") and engine.optimizer is not None:
                    for pg in getattr(engine.optimizer, "param_groups", []):
                        if "max_grad_norm" in pg:
                            pg["max_grad_norm"] = 0.0
                    if hasattr(engine.optimizer, "clip_grad"):
                        try:
                            engine.optimizer.clip_grad = 0.0
                        except Exception:
                            pass
            else:
                # B2) 引擎 config 是对象的情况（老版本/其他分支）
                if getattr(engine, "config", None) is not None:
                    setattr(engine.config, "gradient_clipping", 0.0)

            # C) 运行时强力兜底：彻底短路 DS 的全局 grad-norm 计算与裁剪
            def _noop_scaled_global_norm(self):
                return torch.tensor(0.0, device=getattr(self, "device", None))
            engine.scaled_global_norm = types.MethodType(_noop_scaled_global_norm, engine)

            def _noop_clip_gradients(self, *args, **kwargs):
                return 0.0
            engine.clip_gradients = types.MethodType(_noop_clip_gradients, engine)

    except Exception as e:
        print("disable grad-clip failed:", e)

    # ===== extra DeepSpeed ZeroOptimizer monkeypatch to kill global grad norm all_reduce =====
    try:
        if isinstance(engine, ds.runtime.engine.DeepSpeedEngine):
            zero_opt = getattr(engine, "optimizer", None)

            import types

            # 1) scaled_global_norm() -> constant tensor on correct device
            def _zero_scaled_global_norm(self):
                dev = getattr(self, "device", None)
                if dev is None:
                    # fallback to engine device
                    dev = getattr(engine, "device", None)
                return torch.tensor(0.0, device=dev if dev is not None else "cpu")

            if zero_opt is not None and hasattr(zero_opt, "scaled_global_norm"):
                zero_opt.scaled_global_norm = types.MethodType(_zero_scaled_global_norm, zero_opt)

            # 2) get_grad_norm_direct(...) -> bypass all_reduce entirely
            def _zero_get_grad_norm_direct(self, *args, **kwargs):
                # just return 0.0 on local device, avoid dist.all_reduce
                dev = getattr(self, "device", None)
                if dev is None:
                    dev = getattr(engine, "device", None)
                return torch.tensor(0.0, device=dev if dev is not None else "cpu")

            if zero_opt is not None and hasattr(zero_opt, "get_grad_norm_direct"):
                zero_opt.get_grad_norm_direct = types.MethodType(_zero_get_grad_norm_direct, zero_opt)

            # 3) patch step() INSIDE the zero optimizer, so it doesn't try to
            #    compute global grad norm before stepping
            if zero_opt is not None and hasattr(zero_opt, "step"):
                orig_step = zero_opt.step

                def _zero_step_no_global_norm(*args, **kwargs):
                    # many DS versions call self.scaled_global_norm() inside
                    # so by now that call is monkeypatched to cheap const,
                    # but to be extra safe we just run original step and
                    # swallow any exception that ONLY comes from all_reduce.
                    try:
                        return orig_step(*args, **kwargs)
                    except TypeError as e:
                        # if it *still* hits the ReduceOp incompatible path, skip norm and retry
                        if "AllreduceOptions" in str(e) or "ReduceOp" in str(e):
                            # emulate minimal SGD/AdamW step:
                            # DeepSpeed ZeroOptimizer inherits torch.optim.Optimizer,
                            # and orig_step eventually does param update after norm logic.
                            # If we got here, it's failing *before* finishing update,
                            # so we fallback to a naive param_groups loop.

                            for group in zero_opt.param_groups:
                                lr = group.get("lr", 0.0)
                                if lr == 0.0:
                                    continue
                                beta1, beta2 = group.get("betas", (0.9, 0.999))
                                eps = group.get("eps", 1e-8)
                                weight_decay = group.get("weight_decay", 0.0)

                                # zero_opt may actually wrap DS's FP16Adam-like states,
                                # we try to mimic AdamW-style update.
                                for p in group["params"]:
                                    if p.grad is None:
                                        continue
                                    grad = p.grad
                                    if grad.is_sparse:
                                        raise RuntimeError("sparse grad not supported in fallback")

                                    state = zero_opt.state[p]
                                    if len(state) == 0:
                                        state["exp_avg"] = torch.zeros_like(p.data)
                                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                                        state["step"] = 0

                                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                                    state["step"] += 1
                                    step_t = state["step"]

                                    # Adam update
                                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                                    bias_c1 = 1 - beta1 ** step_t
                                    bias_c2 = 1 - beta2 ** step_t
                                    denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                                    step_size = lr / bias_c1

                                    # AdamW weight decay
                                    if weight_decay != 0:
                                        p.data.add_(p.data, alpha=-weight_decay * lr)

                                    # param update
                                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                            # return something consistent with optimizer.step() semantics
                            return None
                        else:
                            raise

                zero_opt.step = types.MethodType(_zero_step_no_global_norm, zero_opt)

            if rank0: print("[patch] DeepSpeed ZeroOptimizer global grad-norm all_reduce disabled.")

            model._config.checkpoint_config[ds.runtime.model_checkpointing.constants.CHECKPOINT_TAG_VALIDATION] = ds.runtime.model_checkpointing.constants.ValidationMode.IGNORE
            if rank0: print("[patch] DeepSpeed checkpoint tag validation disabled.")

    except Exception as e:
        if rank0: print("[patch] failed to monkeypatch ZeroOptimizer:", e)
    # =====================================================================


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

    with open(args.config_path) as f: config = json.load(f)
    opt = edict(config)
    override = args.opt
    del args.opt
    opt.update(args.__dict__)
    if override is not None:
        for k, v in override:
            try: float(v)  # is number
            except: v = '"' + v + '"'  # is string
            exec('opt.' + k + '=' + v)

    if opt.select_scene is not None and opt.output_dir is None:
        opt.output_dir = './outputs/single/' + f'{opt.select_scene}_' + opt.config_path.split('/')[-1].split('.')[0] # outputs_Slat_vae_full
    elif opt.select_scene is None and opt.output_dir is None:
        opt.output_dir = './outputs/full/' + opt.config_path.split('/')[-1].split('.')[0] + f'_DpMask{opt.use_depth_mask}' # outputs_Slat_vae_full
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
    if os.path.exists(os.path.join(opt.metadata_dir, f'metadata_norm128_v14.csv')):
        metadata = pd.read_csv(os.path.join(opt.metadata_dir, f'metadata_norm128_v14.csv'))
    else: raise ValueError('metadata.csv not found')
    if opt.select_scene is not None:
        metadata = metadata[metadata['sha256'] == opt.select_scene]
        print(f"[INFO] selected scene is {opt.select_scene}, num_metadata: {len(metadata)}")

    print("TTT : ", os.path.exists(os.path.join(opt.metadata_dir, f'less_coverage_scenes_all.csv')))
    less_coverage_csv = pd.read_csv(os.path.join(opt.metadata_dir, f'less_coverage_scenes_all.csv'))
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
    # train_metadata = train_metadata[
    #     train_metadata["sha256"].apply(
    #         lambda sha256: os.path.exists(
    #             os.path.join('/efs-mi-east4-1/z84301856/datasets/scannetpp_v2_opencv_pinhole/features_norm128_v14_fix32/dinov2_vitl14_reg', f'{sha256}.pt')
    #         )
    #     )
    # ]  # 检查过 856 个.pt 文件都存在
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
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True,
                                  persistent_workers=True, drop_last=True,
                                  collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True,
                                persistent_workers=True, collate_fn=collate_skip_none)
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
        train_dataset = DataClass(metadata, opt, stage='train')
        val_dataset = DataClass(metadata, opt, stage='val')
        train_loader = DataLoader(train_dataset, batch_size=opt.trainer.batch_size, 
                                  shuffle=True, num_workers=8, pin_memory=True, 
                                  persistent_workers=True, collate_fn=collate_skip_none)
        val_loader = DataLoader(val_dataset, batch_size=opt.trainer.batch_size, 
                                shuffle=False, num_workers=8, pin_memory=True, 
                                persistent_workers=True, collate_fn=collate_skip_none)        

    # ======================================================
    # 3. Build model
    # ======================================================

    model = ModelClass(opt).bfloat16()
    optimizer, scheduler = model.configure_optimizers()
    optimizer, scheduler = optimizer[0], scheduler[0]

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    # patch_deepspeed(accelerator, model)
    accelerator.register_for_checkpointing(model, optimizer, scheduler)

    if opt.test:
        prepared_test_loaders = []
        for test_loader_idx, test_loader in enumerate(test_loaders):
            testloader_prep = accelerator.prepare_data_loader(test_loader, device_placement=False)
            prepared_test_loaders.append((test_loader_idx, testloader_prep))
        test_loaders = prepared_test_loaders

    epoch = 0
    ckpts = sorted(glob(f'{opt.output_dir}/ckpt/epoch_????.ckpt'))
    if ckpts:
        try:
            ckpt = torch.load(ckpts[-1], accelerator.device)
            epoch = int(ckpts[-1][-9:-5])
        except:
            try:
                ckpt = torch.load(ckpts[-2], accelerator.device)
                epoch = int(ckpts[-2][-9:-5])
            except: ckpt = None
        if ckpt is not None:
            model.load_state_dict(ckpt[0])
            optimizer.load_state_dict(ckpt[1])
            scheduler.load_state_dict(ckpt[2])
    logger.info(f"[INFO] Training starts at epoch {epoch}")

    # =======================================================
    # 4. Training Loop
    # =======================================================

    num_steps_per_epoch = len(train_loader)
    num_steps_per_epoch_val = len(val_loader)
    log_step = epoch * num_steps_per_epoch
    if rank0:
        logger.info(f"[INFO] Training for {opt.trainer.max_epochs} epochs with {num_steps_per_epoch} steps per epoch")
        logger.info(f"[INFO] Evaluating every {opt.trainer.ckpt_every} interval with {num_steps_per_epoch_val} steps per epoch")

    if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.on_train_start()
    else: model.on_train_start()

    for epoch in range(epoch, int(opt.trainer.max_epochs)):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel): model.module.current_epoch = epoch
        else: model.current_epoch = epoch
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
            if rank0 and not ((epoch + 1) % int(opt.trainer.save_img_every_epoch)):
                state_dicts = model.state_dict(), optimizer.state_dict(), scheduler.state_dict()
                torch.save(state_dicts, os.path.join(f'{opt.output_dir}/ckpt/epoch_{epoch + 1:04}.ckpt'))

            # 4.3 Evaluation
            if opt.trainer.eval_every > 0 and (epoch % opt.trainer.eval_every == 0):
                if rank0: logger.info(f"Evaluating on epoch {epoch} global_step {log_step}......")
                model.eval()
                with torch.inference_mode():
                    val_loader_iter = iter(val_loader)
                    val_size = torch.tensor(num_steps_per_epoch_val, device=accelerator.device)
                    sum_val_dict = defaultdict(float)
                    with trange(num_steps_per_epoch_val, desc=f"Validation Epoch {epoch}", disable=not rank0) as pbar_val:
                        for _ in pbar_val:
                            input_dict = next(val_loader_iter)
                            input_dict = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                            _, log_val_dict = model(input_dict, mode='val')
                            for k, v in log_val_dict.items():
                                sum_val_dict[k] += v
                                log_val_dict[k] = v.item()
                            if rank0: pbar_val.set_postfix(log_val_dict)

                val_size = accelerator.reduce(val_size, 'sum')
                sum_val_dict = accelerator.reduce(dict(sum_val_dict), 'sum') 
                if rank0:
                    sum_val_dict = {k: v.item() / val_size for k, v in sum_val_dict.items()}
                    accelerator.log(sum_val_dict, log_step)


# class FilteredIO:
#     _supperss_new_line = False
#     def __init__(self, stream: TextIO, pattern: str): self.stream, self.pattern = stream, pattern
#     def write(self, s: str):
#         if s == self.pattern: self._supperss_new_line = True
#         elif self._supperss_new_line: self._supperss_new_line = False
#         else: self.stream.write(s)
#     def flush(self): self.stream.flush()
if __name__ == '__main__':
    # f_out = FilteredIO(stdout, 'npu gs_sort kernel donot support, using cpu radix sort instead!')
    # with redirect_stdout(f_out):
        # torch.set_anomaly_enabled(True)
        # torch.autograd.set_detect_anomaly(True)
        main()