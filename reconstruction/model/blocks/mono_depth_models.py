import torch
import torch.nn as nn

from utils.platform import DEVICE_NAME

class MonoDepthModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.name == "DepthAnythingV2":
            from submodules.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
            self.depth_model = DepthAnythingV2(**{'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]})
            self.depth_model.load_state_dict(torch.load(self.cfg.ckpt_path.DepthAnythingV2, map_location='cpu'))
        elif self.cfg.name == "UniDepth":
            from submodules.UniDepth.unidepth.models import UniDepthV2
            self.depth_model = UniDepthV2.from_pretrained(self.cfg.ckpt_path.UniDepth)
        elif self.cfg.name == "DepthPro":
            import depth_pro
            self.depth_model, self.transform = depth_pro.create_model_and_transforms(ckpt_path=self.cfg.ckpt_path.DepthPro, device=DEVICE_NAME)
        elif self.cfg.name == "DepthAnythingV3":
            from depth_anything_3.api import DepthAnything3
            self.depth_model = DepthAnything3.from_pretrained(self.cfg.ckpt_path.DepthAnythingV3).to(DEVICE_NAME)
        elif self.cfg.name == "MoGe2":
            from moge.model.v2 import MoGeModel
            self.depth_model = MoGeModel.from_pretrained(self.cfg.ckpt_path.MoGe2)
            self.depth_model = self.depth_model.to(DEVICE_NAME)
        else:
            raise NotImplementedError(f"Depth model type {self.cfg.name} is not implemented.")
        self.depth_model.requires_grad_(False) # Freeze pre-trained depth model.
    
    def infer(self, imgs, f_px=None):
        B, V, C, H, W = imgs.shape
        depth_model_input = imgs.reshape(-1, C, H, W)
        depth_model_output, depth_model_output_mask, intrinsics = None, None, None
        if self.cfg.name == "DepthAnythingV2":
            depth_model_input = (depth_model_input - torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[None, :, None, None]) / \
                torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[None, :, None, None]
            depth_model_output = 1 / (self.depth_model._forward(depth_model_input) + 1e-4)
        elif self.cfg.name == "UniDepth":
            depth_model_output = self.depth_model.infer(depth_model_input)["depth"]
        elif self.cfg.name == "DepthPro":
            depth_model_input = self.transform(depth_model_input)
            depth_model_output = self.depth_model.infer(depth_model_input, f_px=f_px)["depth"]
        elif self.cfg.name == "DepthAnythingV3":
            depth_model_input = (depth_model_input.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            numpy_list = [sample for sample in depth_model_input]
            prediction = self.depth_model.inference(numpy_list)
            depth_model_output = torch.from_numpy(prediction.depth)
        elif self.cfg.name == "MoGe2":
            predictions = [self.depth_model.infer(ipt) for ipt in depth_model_input]
            depth_model_output = torch.stack([pred["depth"].unsqueeze(0) for pred in predictions], dim=0)
            depth_model_output_mask = torch.stack([pred["mask"].unsqueeze(0) for pred in predictions], dim=0)
            intrinsics = torch.stack([pred["intrinsics"] for pred in predictions], dim=0)
        else:
            raise NotImplementedError(f"Depth model type {self.ckpt_depth_model.name} is not implemented.")
        return depth_model_output, depth_model_output_mask, intrinsics
        