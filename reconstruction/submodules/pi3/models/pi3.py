import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg, dinov2_vitb14_reg
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint

from utils.platform import DEVICE_NAME
import utils.common as cm

import os
USE_CHECKPOINT = os.environ.get('USE_CHECKPOINT', True)

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            cfg, 
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        if cfg.decoder_size == "giant":
            self.encoder = dinov2_vitg14_reg(pretrained=True, model_dir="ckpt/dinov2-giant")
        elif cfg.decoder_size == "large" or cfg.decoder_size == "large-double":
            self.encoder = dinov2_vitl14_reg(pretrained=True, model_dir="ckpt/dinov2-large")
        elif cfg.decoder_size == "base":
            self.encoder = dinov2_vitb14_reg(pretrained=True, model_dir="ckpt/dinov2-base")
        elif cfg.decoder_size == "small":
            self.encoder = dinov2_vits14_reg(pretrained=True, model_dir="ckpt/dinov2-small")
        else:
            raise NotImplementedError("Not implemented decode size: '{}'".format(cfg.decoder_size))
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = cfg.pos_type if cfg.pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        head_depth = 5
        if cfg.decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif cfg.decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif cfg.decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        elif cfg.decoder_size == 'large-double':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 72
            head_depth = 10
        elif cfg.decoder_size == 'giant':
            dec_embed_dim = 1536
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 72
            head_depth = 10
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        assert not (cfg.point_head.enabled and cfg.depth_head.enabled), "Only one of point head and depth head can be enabled, not both."
        assert cfg.point_head.enabled or cfg.depth_head.enabled, "Must enabled one of point head and depth head."

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        if cfg.point_head.enabled:
            self.point_decoder = TransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
                depth=head_depth,
                use_checkpoint=USE_CHECKPOINT
            )
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, activation="linear")
            self.xy_decoder = TransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
                depth=head_depth,
                use_checkpoint=USE_CHECKPOINT
            )
            self.xy_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=2, activation="linear")

        # ----------------------
        #  Local Depths Decoder
        # ----------------------
        if cfg.depth_head.enabled:
            self.depth_decoder = TransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
                depth=head_depth,
                use_checkpoint=USE_CHECKPOINT
            )
            self.depth_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, activation="exp")

        # ----------------------
        #     Conf Decoder
        # ----------------------
        if hasattr(self, "point_head"):
            self.conf_decoder = deepcopy(self.point_decoder)
            # self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, activation="linear")
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, activation="2sigmoid")
        elif hasattr(self, "depth_head"):
            self.conf_decoder = deepcopy(self.depth_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, activation="2sigmoid")
        else:
            raise NotImplementedError()

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            depth=head_depth,
            use_checkpoint=USE_CHECKPOINT
        )
        self.camera_head = CameraHead(dim=512, pred_fov=hasattr(self, "depth_head"))

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W, xpos_max=None):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            if self.training and USE_CHECKPOINT:
                hidden = checkpoint(blk, hidden, pos, xpos_max, use_reentrant=True)
            else:
                hidden = blk(hidden, xpos=pos, xpos_max=xpos_max)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def _forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        xpos_max = max(H//self.patch_size, W//self.patch_size)
        hidden, pos = self.decode(hidden, N, H, W, xpos_max=xpos_max)

        if hasattr(self, "point_head"):
            point_hidden = self.point_decoder(hidden, xpos=pos, xpos_max=xpos_max)
            xy_hidden = self.xy_decoder(hidden, xpos=pos, xpos_max=xpos_max)
        else:
            point_hidden = self.depth_decoder(hidden, xpos=pos, xpos_max=xpos_max)
        conf_hidden = self.conf_decoder(hidden, xpos=pos, xpos_max=xpos_max)
        camera_hidden = self.camera_decoder(hidden, xpos=pos, xpos_max=xpos_max)

        with torch.amp.autocast(device_type=DEVICE_NAME, enabled=False):
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w)
            if hasattr(self, "depth_head"):
                camera_poses, camera_fovs = camera_poses
                camera_poses = camera_poses.reshape(B, N, 4, 4)
                camera_fovs = camera_fovs.reshape(B, N, 2)
                camera_intrs = cm.fov2intr(camera_fovs) # b n 3 3
                xy = cm.intr2ray(camera_intrs, H, W)[..., :2, :, :] # b n 2 h w
                xy = xy.permute(0, 1, 3, 4, 2) # b n h w 2
            else:
                camera_poses = camera_poses.reshape(B, N, 4, 4)
            
            # local points
            point_hidden = point_hidden.float()
            if hasattr(self, "point_head"):
                xy_hidden = xy_hidden.float()
                xy = self.xy_head([xy_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                z = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                z = torch.exp(z)
            else:
                z = self.depth_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            local_points = torch.cat([xy, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

        return dict(
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
    
    def forward(self, imgs):
        return self._forward(imgs)
