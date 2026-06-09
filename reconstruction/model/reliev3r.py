import os
import torch
import torch.nn as nn
from einops import rearrange

from submodules.pi3.models.pi3 import Pi3
from utils.common import normalize_trans, pad_one, calc_extr_inv


class Reliev3R(nn.Module):
    def __init__(self, cfg):
        """Initialization.
        """
        super().__init__()
        self.cfg = cfg
        
        # Initialize depth-aligner and cross-predictor
        if cfg.depth_aligner.type == "pi3":
            self.depth_aligner = Pi3(cfg=cfg.depth_aligner.pi3)
            self.pred_z_depth = (os.getenv("pred_z_depth", "False") == "True")
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(cfg.depth_aligner.type))
    
    def _forward(self, data):
        img = data["input_images"]

        aligned_depth_dict = self.infer_aligned_depth(img, data["input_depths"])
        if self.cfg.depth_aligner.type == "pi3":
            cameras = aligned_depth_dict["camera_poses"]
            if self.cfg.decoder.camera_decoder.get("sigmoid_trans", False):
                cameras_tmp = cameras.clone()
                cameras_tmp[..., :3, 3] = torch.nn.functional.sigmoid(cameras[..., :3, 3]) * 2 - 1
                cameras = cameras_tmp.clone()
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
                # "depths_conf":            aligned_depth_dict["confs"],      # Predicted depth confidence, b v 1 h w
                "camera_extrs":           cameras,                          # Predicted camera poses, b v 4 4.
            }
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(self.cfg.depth_aligner.type))
        
        return output_dict
    
    def _forward_eval(self, data):
        img       = data["input_images"]

        B, V, C, H, W = img.shape
        pe_helper_global, pe_helper_local =  (None, None)
        aligned_depth_dict = self.infer_aligned_depth(img, data["input_depths"], 
                                                      pe_helper_global, pe_helper_local, f_px=None)
        
        if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
            if self.cfg.depth_aligner.type == "ours":
                camera_tokens0 = self.camera_decoder(rearrange(aligned_depth_dict["aligned_depth_tokens"][-1], "b (v l) d -> (b v) l d", v=V))
                assert H % 14 == 0 and W % 14 == 0
                cameras0 = self.camera_head(camera_tokens0, patch_h=H//14, patch_w=W//14, v=V).reshape(B, V, 4, 4)
            elif self.cfg.depth_aligner.type == "pi3":
                cameras0 = aligned_depth_dict["camera_poses"]
                camera_fovs = aligned_depth_dict["camera_fovs"]
            else:
                raise NotImplementedError("Not implemented depth aligner type '{}'".format(self.cfg.depth_aligner.type))
            if self.cfg.decoder.camera_decoder.get("normalize_trans", False):
                # print("before normalization: {}".format(cameras0.norm(dim=-1).mean(dim=-1).mean().item()))
                cameras0 = normalize_trans(cameras0, detach_norm_factor=False)
                # print("after normalization: {}".format(cameras0.norm(dim=-1).mean(dim=-1).mean().item()))
        
        output_dict = {
            "mono_depths":      aligned_depth_dict["mono_depth"],         # Pseudo depth label
            "depths":           aligned_depth_dict["aligned_depth"],      # Depth prediction
            # "depths_conf":      aligned_depth_dict["aligned_depth_conf"], # Predicted depth confidence
            "scaled_depths":    aligned_depth_dict["scaled_depth"],       # Scaled depth prediction
            "pred_depth_scale": aligned_depth_dict["depth_scales"],       # Predicted depth scale
            "pred_depth_trans": aligned_depth_dict["depth_transs"],       # Predicted depth transition
        }
        if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
            output_dict.update({
                "camera_extrs": cameras0, # Camera pose prediction
                "camera_fovs":  camera_fovs, # Camera field of view along x-axis and y-axis, tensor of shape b,v,2
            })
        
        return output_dict
    
    def forward(self, data):
        """Training time forward pass.

        Args:
            data (dict): Return from dataset.
        """
        # if self.training:
        #     return self._forward(data)
        # else:
        #     return self._forward_eval(data)
        
        return self._forward(data)
    
    def infer_aligned_depth(self, imgs, pseudo_depth):
        B, V, C, H, W = imgs.shape
        output_dict = {}
        output_dict["mono_depth"] = pseudo_depth.reshape(B, V, 1, H, W)
        if self.cfg.depth_aligner.type == "pi3":
            depth_aligner_outputs = self.depth_aligner(imgs)
            output_dict.update({
                "aligner_output": rearrange(depth_aligner_outputs["local_points"], "b v h w c -> b v c h w"), # b v 3 h w
                # "confs":          rearrange(depth_aligner_outputs["conf"], "b v h w c -> b v c h w"), # b v 1 h w
                "camera_poses":   depth_aligner_outputs["camera_poses"], # b v 4 4
            })
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(self.cfg.depth_aligner.type))
        return output_dict


class Reliev3R_Pi3(Pi3):
    def __init__(self, cfg):
        """Initialization.
        """
        super().__init__(cfg=cfg.depth_aligner.pi3)
        self.cfg = cfg
        self.pred_z_depth = (os.getenv("pred_z_depth", "False") == "True")
        self.pred_camera_is_w2c = cfg.depth_aligner.pi3.get("camera_conversion", "w2c") == "w2c"
    
    def _forward(self, data):
        img = data["input_images"]

        aligned_depth_dict = self.infer_aligned_depth(img, data["input_depths"])
        if self.cfg.depth_aligner.type == "pi3":
            cameras = aligned_depth_dict["camera_poses"]
            if self.cfg.decoder.camera_decoder.get("sigmoid_trans", False):
                cameras_tmp = cameras.clone()
                cameras_tmp[..., :3, 3] = torch.nn.functional.sigmoid(cameras[..., :3, 3]) * 2 - 1
                cameras = cameras_tmp.clone()
            if self.cfg.decoder.camera_decoder.get("normalize_trans", False):
                cameras = normalize_trans(cameras, detach_norm_factor=False)
            if not self.pred_camera_is_w2c: 
                # Because dataset camera is in w2c format, we should transform model output camera to w2c.
                cameras = calc_extr_inv(cameras)
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
                # "depths_conf":            aligned_depth_dict["confs"] + 1e-3, # Predicted depth confidence, b v 1 h w
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
            depth_aligner_outputs = super()._forward(imgs)
            output_dict.update({
                "aligner_output": rearrange(depth_aligner_outputs["local_points"], "b v h w c -> b v c h w"), # b v 3 h w
                # "confs":          rearrange(depth_aligner_outputs["conf"], "b v h w c -> b v c h w"), # b v 1 h w
                "camera_poses":   depth_aligner_outputs["camera_poses"], # b v 4 4
            })
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(self.cfg.depth_aligner.type))
        return output_dict
