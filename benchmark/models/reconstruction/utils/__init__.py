import torch
from . import config_loader, image_utils
from .platform import PLATFORM, DEVICE_NAME
from .loss_utils import LossFunction
from submodules.pi3.models.loss import Pi3Loss
from utils.common import calc_extr_inv
from einops import rearrange

__all__ = [config_loader, image_utils, PLATFORM, DEVICE_NAME]

class Pi3LossWrapper(Pi3Loss):
    """Transform the input data and model output of ours into Pi3 style.
    """
    
    def __init__(self, train_conf=False):
        super().__init__(train_conf)
    
    def forward(self, output_dict: dict[str, torch.Tensor], data: dict[str, torch.Tensor]):
        B, V = data["input_images"].shape[:2]
                
        gt_raw_pi3 = [dict(img=data["input_images"][:, v], 
                           pts3d=data["input_points"][:, v], 
                           valid_mask=data["input_masks"][:, v].squeeze(-3), 
                           camera_pose=calc_extr_inv(data["input_viewmats"][:, v]), # transform our w2c pose to c2w pose.
                           ) for v in range(V)]
        pred_pi3 = dict(local_points=rearrange(output_dict["depths"], "b v c h w -> b v h w c"), 
                        camera_poses=calc_extr_inv(output_dict["camera_extrs"]), # We define the model output as w2c trans to fit our codebase.
                        conf=rearrange(output_dict["depths_conf"], "b v c h w -> b v h w c"), 
                        )
        
        final_loss, details = super()._forward(pred_pi3, gt_raw_pi3)
        details.update({"total": final_loss})
        return details

def make_loss_func(cfg, matching_model=None):
    if cfg.name == "ours":
        return LossFunction(cfg, matching_model=matching_model)
    elif cfg.name == "pi3":
        return Pi3LossWrapper(train_conf=True)
    else:
        raise NotImplementedError("Loss type {} is not implemented.".format(cfg.name))