from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
from datasets.base.base_dataset import sample_resolutions
import hydra

from pi3.models.loss import Pi3Loss
from utils.eval_utils import eval_angle, eval_trajectory, eval_depth_map
from pi3.utils.geometry import se3_inverse

class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.train_loss)

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            encoder_params = [param for param in model_.encoder.named_parameters()]
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if not name.startswith("encoder.") and not '.encoder.' in name
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                return [
                    {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
                    {"params": decay, "weight_decay": weight_decay, 'lr': lr},
                ]

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        # if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
        #     self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        # if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
        #     self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        # if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
        #     self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        # if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
        #     self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        # if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
        #     self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        # if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
        #     self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        # if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
        #     self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        # ---------- train ----------
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            # custom dataset, can pass base_seed
            try:
                self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
            except TypeError:
                self.train_loader.dataset.set_epoch(epoch)

        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            # PyTorch DistributedSampler 只能传 epoch
            self.train_loader.sampler.set_epoch(epoch)

        # handle accelerate wrapped dataloader
        if (
            hasattr(self.train_loader, 'batch_sampler')
            and hasattr(self.train_loader.batch_sampler, 'batch_sampler')
            and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler')
            and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch')
        ):
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch)

        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):
            try:
                self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
            except TypeError:
                self.train_loader.batch_sampler.set_epoch(epoch)

        # ---------- test ----------
        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            # 验证集通常固定即可；0
            try:
                self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
            except TypeError:
                self.test_loader.dataset.set_epoch(0)

        if (
            hasattr(self.test_loader, 'batch_sampler')
            and hasattr(self.test_loader.batch_sampler, 'batch_sampler')
            and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler')
            and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch')
        ):
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch)

        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'set_epoch'):
            try:
                self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
            except TypeError:
                self.test_loader.batch_sampler.set_epoch(epoch)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)
            
    def forward_batch(self, batch, mode='train'):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        pred = self.model(imgs)

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

        return EasyDict(
            loss=loss,
            **details
        )

    def calculate_metrics(self, output, batch, mode='test'):
        if mode != 'test':
            return {}

        pred, gt_raw = output
        metrics = {}

        try:
            device = pred['local_points'].device

            gt = self.test_loss.prepare_gt(gt_raw, device)
            # 避免 inplace 改原始 pred，做一个浅拷贝 + tensor clone
            pred_norm = EasyDict()
            for k, v in pred.items():
                pred_norm[k] = v.clone() if torch.is_tensor(v) else v

            pred_norm = self.test_loss.normalize_pred(pred_norm, gt)
            pred_pose = pred_norm["camera_poses"]                  # [B, N, 4, 4], c2w
            pred_w2c0 = se3_inverse(pred_pose[:, 0])              # [B, 4, 4]
            pred_norm["camera_poses"] = torch.einsum(
                "bij,bnjk->bnik", pred_w2c0, pred_pose
            )
            # ---------------------------
            # camera metrics
            # ---------------------------
            pr_cam_norm = pred_norm["camera_poses"][0].detach()
            gt_cam_norm = gt["camera_poses"][0]
            # relative-angle metrics and trajectory metrics are independent: a failure in one must not drop the other
            try:
                rra_acc, rta_acc, auc_at_thre = eval_angle(pr_cam_norm, gt_cam_norm)
                metrics["cam_rra_acc"] = float(rra_acc)
                metrics["cam_rta_acc"] = float(rta_acc)
                metrics["cam_auc_at_thre"] = float(auc_at_thre)
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.log_info(f"[validate] camera angle metric skipped: {e}")
            try:
                ate, rpe_trans, rpe_rot = eval_trajectory(pr_cam_norm, gt_cam_norm)
                metrics["cam_ate"] = float(ate)
                metrics["cam_rpe_trans"] = float(rpe_trans)
                metrics["cam_rpe_rot"] = float(rpe_rot)
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.log_info(f"[validate] camera trajectory metric skipped: {e}")

            # ---------------------------
            # depth metrics
            # 用 normalize 后的 local_points z 来算
            # ---------------------------
            try:
                pr_depth = pred_norm["local_points"][0, ..., 2].unsqueeze(1)   # [N,1,H,W]
                gt_depth = gt["local_points"][0, ..., 2].unsqueeze(1)          # [N,1,H,W]
                gt_mask = gt["valid_masks"][0].unsqueeze(1)                    # [N,1,H,W]

                depth_metrics = eval_depth_map(
                    pr_depth.detach(),
                    gt_depth,
                    gt_mask > 0.0,
                    align_with_lstsq=True,
                    device=str(self.accelerator.device),
                )

                if "Abs Rel" in depth_metrics:
                    metrics["depth_abs_rel"] = float(depth_metrics["Abs Rel"])
                if "δ < 1.25" in depth_metrics:
                    metrics["depth_delta_1.25"] = float(depth_metrics["δ < 1.25"])
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.log_info(f"[validate] depth metric skipped: {e}")

        except Exception as e:
            if self.accelerator.is_main_process:
                self.log_info(f"[validate] metric prepare failed: {e}")

        return metrics

def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
