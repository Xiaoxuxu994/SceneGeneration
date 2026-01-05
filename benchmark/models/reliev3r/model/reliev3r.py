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
from .blocks.mono_depth_models import MonoDepthModel


class Reliev3R(nn.Module):
    def __init__(self, cfg, depth_preprocessed=True):
        """Initialization.
        """
        super().__init__()
        self.cfg = cfg
        
        # Load monocular depth model.
        if not depth_preprocessed:
            self.depth_model = MonoDepthModel(cfg=self.cfg.depth_model)
        # Initialize depth-aligner and cross-predictor
        if cfg.depth_aligner.type == "ours":
            self.depth_aligner = DepthAligner(cfg.depth_aligner.ours)
            if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
                self.camera_decoder = TransformerDecoder(in_dim=cfg.depth_aligner.ours.d_model, 
                                                        out_dim=cfg.depth_aligner.ours.d_model)
                self.camera_head = CameraHead(dim=cfg.depth_aligner.ours.d_model)
        elif cfg.depth_aligner.type == "pi3":
            self.depth_aligner = Pi3(cfg=cfg.depth_aligner.pi3)
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(cfg.depth_aligner.type))
    
    def _forward(self, data):
        img       = data["input_images"]
        viewmats  = data["input_viewmats"]
        Ks        = data["input_Ks"]
        
        B, V, C, H, W = img.shape
        
        if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
            unit_viewmats = torch.eye(4)[None, None].expand_as(viewmats).to(device=DEVICE_NAME)
            if data.get("pseudo_viewmats", None) is not None:
                viewmat_place_holder = data["pseudo_viewmats"].to(device=DEVICE_NAME)
            else:
                viewmat_place_holder = unit_viewmats
            pe_helper_global, pe_helper_local = self.create_pe_helper_for_aligner(W, H, viewmat_place_holder, Ks) \
                if self.cfg.depth_aligner.type == "ours" else (None, None)
        else:
            pe_helper_global, pe_helper_local = self.create_pe_helper_for_aligner(W, H, viewmats, Ks) \
                if self.cfg.depth_aligner.type == "ours" else (None, None)
        aligned_depth_dict = self.infer_aligned_depth(img, data["input_depths"], 
                                                      pe_helper_global, pe_helper_local, f_px=Ks[0, 0, 0, 0])
        
        if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
            if self.cfg.depth_aligner.type == "ours":
                camera_tokens0 = self.camera_decoder(rearrange(aligned_depth_dict["aligned_depth_tokens"][-1], "b (v l) d -> (b v) l d", v=V))
                assert H % 14 == 0 and W % 14 == 0
                cameras0 = self.camera_head(camera_tokens0, patch_h=H//14, patch_w=W//14, v=V).reshape(B, V, 4, 4)
            elif self.cfg.depth_aligner.type == "pi3":
                cameras0 = aligned_depth_dict["camera_poses"]
            else:
                raise NotImplementedError("Not implemented depth aligner type '{}'".format(self.cfg.depth_aligner.type))
        
        output_dict = {
            "mono_depths":      aligned_depth_dict["mono_depth"],         # Pseudo depth label
            "depths":           aligned_depth_dict["aligned_depth"],      # Depth prediction
            "depths_conf":      aligned_depth_dict["aligned_depth_conf"], # Predicted depth confidence
            "scaled_depths":    aligned_depth_dict["scaled_depth"],       # Scaled depth prediction
            "pred_depth_scale": aligned_depth_dict["depth_scales"],       # Predicted depth scale
            "pred_depth_trans": aligned_depth_dict["depth_transs"],       # Predicted depth transition
        }
        if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
            output_dict.update({
                "camera_extrs": cameras0, # Camera pose prediction
            })
        
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
            else:
                raise NotImplementedError("Not implemented depth aligner type '{}'".format(self.cfg.depth_aligner.type))
        
        output_dict = {
            "mono_depths":      aligned_depth_dict["mono_depth"],         # Pseudo depth label
            "depths":           aligned_depth_dict["aligned_depth"],      # Depth prediction
            "depths_conf":      aligned_depth_dict["aligned_depth_conf"], # Predicted depth confidence
            "scaled_depths":    aligned_depth_dict["scaled_depth"],       # Scaled depth prediction
            "pred_depth_scale": aligned_depth_dict["depth_scales"],       # Predicted depth scale
            "pred_depth_trans": aligned_depth_dict["depth_transs"],       # Predicted depth transition
        }
        if self.cfg.get("decoder", None) is not None and self.cfg.decoder.camera_decoder.enabled:
            output_dict.update({
                "camera_extrs": cameras0, # Camera pose prediction
            })
        
        return output_dict
            
    def forward(self, data):
        """Training time forward pass.

        Args:
            data (dict): Return from dataset.
        """
        if self.training:
            return self._forward(data)
        else:
            return self._forward_eval(data)
    
    def create_pe_helper_for_aligner(self, W, H, viewmats, Ks):
        pe_helper_aligner_global = AtomicAttnPEHelper(pe_type="prope", 
                                                      head_dim=self.depth_aligner.cfg.d_model//self.depth_aligner.cfg.n_heads, 
                                                      patches_x=W//self.depth_aligner.cfg.patch_size, 
                                                      patches_y=H//self.depth_aligner.cfg.patch_size, 
                                                      image_width=W, image_height=H, 
                                                      viewmats_tar=viewmats, Ks_tar=Ks, 
                                                      viewmats_src=viewmats, Ks_src=Ks, #) 
                                                      n_cameras=viewmats.shape[1] if self.cfg.depth_aligner.depth_scaler.enabled else None)
        pe_helper_aligner_local = AtomicAttnPEHelper(pe_type="prope", 
                                                     head_dim=self.depth_aligner.cfg.d_model//self.depth_aligner.cfg.n_heads, 
                                                     patches_x=W//self.depth_aligner.cfg.patch_size, 
                                                     patches_y=H//self.depth_aligner.cfg.patch_size, 
                                                     image_width=W, image_height=H, 
                                                     viewmats_tar=viewmats.view(-1, 1, 4, 4), Ks_tar=Ks.view(-1, 1, 3, 3), 
                                                     viewmats_src=viewmats.view(-1, 1, 4, 4), Ks_src=Ks.view(-1, 1, 3, 3), #)
                                                     n_cameras=1 if self.cfg.depth_aligner.depth_scaler.enabled else None)
        return pe_helper_aligner_global, pe_helper_aligner_local
    
    def infer_aligned_depth(self, imgs, pseudo_depth=None, pe_helper_global=None, pe_helper_local=None, f_px=None):
        B, V, C, H, W = imgs.shape
        output_dict = {}
        # with torch.no_grad():
        #     depth_model_output = self.depth_model.infer(imgs, f_px=f_px) if pseudo_depth is None else pseudo_depth # (BxV, 1, H, W)
        # output_dict["mono_depth"] = depth_model_output.reshape(B, V, 1, H, W)
        print("output_dict-mono_depth = None")
        output_dict["mono_depth"] = None
        
        if self.cfg.depth_aligner.type == "ours":
            depth_aligner_outputs = self.depth_aligner(imgs, pe_helper_global=pe_helper_global, pe_helper_local=pe_helper_local)
            output_dict.update({
                "aligned_depth":        depth_aligner_outputs["aligned_depths"].reshape(B, V, 1, H, W), 
                "aligned_depth_tokens": depth_aligner_outputs["aligned_depths_tokens"], 
                "aligned_depth_conf":   depth_aligner_outputs["aligned_depths_conf"].reshape(B, V, 1, H, W), 
                "scaled_depth":         depth_aligner_outputs["scaled_depths"].reshape(B, V, 1, H, W), 
                "depth_scales":         depth_aligner_outputs["depth_scales"], 
                "depth_transs":         depth_aligner_outputs["depth_transs"], 
            })
        elif self.cfg.depth_aligner.type == "pi3":
            depth_aligner_outputs = self.depth_aligner(imgs)
            output_dict.update({
                "aligned_depth":        rearrange(depth_aligner_outputs["local_points"], "b v h w c -> b v c h w"), 
                "aligned_depth_tokens": None, 
                "aligned_depth_conf":   rearrange(depth_aligner_outputs["conf"], "b v h w c -> b v c h w"), 
                "scaled_depth":         rearrange(depth_aligner_outputs["local_points"], "b v h w c -> b v c h w"), 
                "depth_scales":         None, 
                "depth_transs":         None, 
                "camera_poses":         depth_aligner_outputs["camera_poses"], 
            })
        else:
            raise NotImplementedError("Not implemented depth aligner type: {}".format(self.cfg.depth_aligner.type))
        return output_dict

    def enable_checkpoint(self, 
                          enc_skip_first=4,
                          enc_skip_last=4,
                          dec_skip_first=4,
                          dec_skip_last=3):
        def _apply_ckpt(module):
            # safe enable checkpoint on each module
            if hasattr(module, "_orig_forward"):
                return 
            module._orig_forward = module.forward
            def forward(*args, module=module, **kwargs):
                if module.training:
                    return checkpoint(
                        module._orig_forward,
                        *args,
                        use_reentrant=False,
                        **kwargs
                    )
                return module._orig_forward(*args, **kwargs)
            module.forward = forward

        # DinoV2 encoder blocks
        enc_blocks = self.depth_aligner.encoder.blocks
        n_enc = len(enc_blocks)
        print(f"encoder has {n_enc} layers")
        start_enc = min(enc_skip_first, n_enc)
        end_enc = max(start_enc, n_enc - enc_skip_last)
        for i in range(start_enc, end_enc):
            _apply_ckpt(enc_blocks[i])

        # Decoder blocks
        dec_blocks = self.depth_aligner.decoder
        n_dec = len(dec_blocks)
        print(f"decoder has {n_dec} layers")
        start_dec = min(dec_skip_first, n_dec)
        end_dec = max(start_dec, n_dec - dec_skip_last)
        for i in range(start_dec, end_dec):
            _apply_ckpt(dec_blocks[i])