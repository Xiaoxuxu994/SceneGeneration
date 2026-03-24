import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from submodules.pi3.models.pi3 import Pi3
from submodules.pi3.models.layers.transformer_head import TransformerDecoder
from submodules.pi3.models.layers.camera_head import CameraHead
from .depth_aligner import DepthAligner
from utils.platform import DEVICE_NAME
from utils.embedder import AtomicAttnPEHelper
from utils.common import normalize_trans, pad_one
from .blocks.mono_depth_models import MonoDepthModel


class Reliev3R(nn.Module):
    def __init__(self, cfg, depth_preprocessed=True):
        """Initialization.
        """
        super().__init__()
        self.cfg = cfg
        self.register_buffer('eye4', torch.eye(4))
        
        # Load monocular depth model.
        if not depth_preprocessed:
            self.depth_model = MonoDepthModel(cfg=self.cfg.depth_model)
        # Initialize depth-aligner and cross-predictor
        if cfg.depth_aligner.type == "pi3":
            self.depth_aligner = Pi3(cfg=cfg.depth_aligner.pi3)
            self.pred_z_depth = cfg.depth_aligner.pi3.get("pred_z_depth", True)
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(cfg.depth_aligner.type))
    
    def _forward(self, data):
        img = data["input_images"]

        aligned_depth_dict = self.infer_aligned_depth(img, data["input_depths"])
        if self.cfg.depth_aligner.type == "pi3":
            cameras = aligned_depth_dict["camera_poses"]
            if self.cfg.decoder.camera_decoder.get("normalize_trans", False):
                cameras = normalize_trans(cameras, detach_norm_factor=False)
            aligner_output = aligned_depth_dict["aligner_output"] # b v 3 h w
            cam_rays = pad_one(aligner_output[..., :2, :, :], dim=-3) # b v 3 h w
            pred_depths = aligner_output[..., -1:, :, :] # b v 1 h w
            if not self.pred_z_depth:
                cam_rays = cam_rays / cam_rays.norm(dim=-3, keepdim=True)
            output_dict = {
                "mono_depth":             aligned_depth_dict["mono_depth"], # Pseudo depth label, b v 1 h w
                "local_points":           cam_rays * pred_depths,           # Local points, b v 3 h w
                "local_points_detach_xy": cam_rays.detach() * pred_depths,  # Local points, intrs detached, b v 3 h w
                "local_points_detach_z":  cam_rays * pred_depths.detach(),  # Local points, depths detached, b v 3 h w
                "depths":                 pred_depths,                      # Predicted depth, b v 1 h w
                "depths_conf":            aligned_depth_dict["confs"],      # Predicted depth confidence, b v 1 h w
                "camera_extrs":           cameras,                          # Predicted camera poses, b v 4 4.
            }
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(self.cfg.depth_aligner.type))
        
        return output_dict
        
    def forward(self, data):
        """Training time forward pass.

        Args:
            data (dict): Return from dataset.
        """
        return self._forward(data)
    
    def infer_aligned_depth(self, imgs, pseudo_depth):
        B, V, C, H, W = imgs.shape
        output_dict = {}
        output_dict["mono_depth"] = pseudo_depth.reshape(B, V, 1, H, W)
        if self.cfg.depth_aligner.type == "pi3":
            depth_aligner_outputs = self.depth_aligner(imgs)
            output_dict.update({
                "aligner_output": rearrange(depth_aligner_outputs["local_points"], "b v h w c -> b v c h w"), # b v 3 h w
                "confs":          rearrange(depth_aligner_outputs["conf"], "b v h w c -> b v c h w"), # b v 1 h w
                "camera_poses":   depth_aligner_outputs["camera_poses"], # b v 4 4
            })
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(self.cfg.depth_aligner.type))
        return output_dict
