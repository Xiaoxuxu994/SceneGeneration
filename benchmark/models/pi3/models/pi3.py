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
from .dinov2.hub.backbones import (
    dinov2_vitl14, 
    dinov2_vitl14_reg, 
    dinov2_vitb14_reg, 
    dinov2_vits14_reg,
    dinov2_vitg14_reg)
from .dinov3.hub.backbones import (
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitb16,
    dinov3_vits16plus,
    dinov3_vits16,    
)
from huggingface_hub import PyTorchModelHubMixin

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            model_size='dinov3_huge+_3Bv2',
        ):
        super().__init__()

       # -------------------------------------------
        #        Resolve Encoder / decoder spec
        # -------------------------------------------
        enc_type, encoder_size, model_variant = self._resolve_model_spec(
            decoder_size=decoder_size,
            model_size=model_size
        )

        backbone_cfg = self._build_backbone_cfg()

        if enc_type not in backbone_cfg:
            raise NotImplementedError(f"Not implemented enc_type: {enc_type}")
        if encoder_size not in backbone_cfg[enc_type]:
            raise NotImplementedError(f"Not implemented encoder_size: {enc_type}_{encoder_size}")
        
        spec = backbone_cfg[enc_type][encoder_size]

        # ----------------------
        #        Encoder
        # ----------------------       
        self.encoder = spec['builder'](pretrained=True, model_dir=spec['model_dir'])
        self.patch_size = getattr(self.encoder,"patch_size",14)
        if enc_type == 'dinov2' and hasattr(self.encoder,'mask_token'):
            del self.encoder.mask_token
        if enc_type == 'dinov3' and hasattr(self.encoder,'mask_token'):
            del self.encoder.mask_token
        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # -------------------------------------------
        #        Decoder Spec
        # -------------------------------------------        
        if model_variant is not None:
            if model_variant in spec['variants']:
                dec_cfg = spec['variants'][model_variant]
            elif spec['default_decoder'] is not None:
                dec_cfg = spec['default_decoder']
            else:
                raise NotImplementedError(
                    f"Not implemented model_size: '{enc_type}_{encoder_size}_{model_variant}"
                )
        else:
            if spec['default_decoder'] is None:
                raise NotImplementedError(
                    f"Please set model_size explicitly for '{enc_type}_{encoder_size}"
                )
            dec_cfg = spec['default_decoder']
        
        dec_embed_dim = dec_cfg['dec_embed_dim']
        dec_num_heads = dec_cfg['dec_num_heads']
        mlp_ratio =  dec_cfg['mlp_ratio']
        dec_depth = dec_cfg['dec_depth']     
        head_depth = dec_cfg['head_depth']   

        # -----------------------------------
        #        Projec & Norm Layer
        # -----------------------------------
        self.enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features
        self.dec_embed_dim = dec_embed_dim
        self.project_e2d =(
            nn.Linear(self.enc_embed_dim,self.dec_embed_dim,bias=False)
            if self.enc_embed_dim != self.dec_embed_dim else nn.Identity()
        )
        self.project_norm=(
            nn.LayerNorm(self.dec_embed_dim)
            if self.enc_embed_dim != self.dec_embed_dim else nn.Identity()
        )
        # ----------------------
        #        Decoder
        # ----------------------
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=self.dec_embed_dim,
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

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            depth=head_depth,
        )
        self.point_head = LinearPts3d(patch_size=self.patch_size, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=self.patch_size, dec_embed_dim=1024, output_dim=1)

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
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @staticmethod
    def _resolve_model_spec(decoder_size,model_size):
        """
            Re-construct based on zhaobo's script
            Resolve: enc_type,encoder_size,model_variant
            
            Examples:
                decoder_size='large'->('dinov2','large','0.95B')
                model_size='dinov3_large_2.81B->('dinov3','large','2.81B')
        """
        if model_size is not None:
            parts = model_size.split('_')
            if len(parts) != 3:
                raise ValueError(
                    f"model_size should look like 'dinov3_large_2.81B', got: {model_size}"
                )
            enc_type, encoder_size, model_variant = parts
            return enc_type, encoder_size, model_variant
        
        # backward-compatitble old interface
        enc_type = 'dinov2'
        if decoder_size == 'small':
            return enc_type, 'small', None
        if decoder_size == 'base':
            return enc_type, 'base', None
        if decoder_size == 'large':
            return enc_type, 'large', '0.95B'
        else:
            raise NotImplementedError(f"Note implemented decoder_size: '{decoder_size}'")

    @staticmethod
    def _build_backbone_cfg():
        return {
            'dinov2':{
                'small':{
                    'builder': dinov2_vits14_reg,
                    'model_dir': 'ckpts/dinov2-small',
                    'default_decoder': dict(
                        dec_embed_dim=384,
                        dec_num_heads=6,
                        mlp_ratio=4,
                        dec_depth=24,
                        head_depth=5,
                    ),
                    'variants': {}
                },
                'base':{
                    'builder': dinov2_vitb14_reg,
                    'model_dir': 'ckpts/dinov2-base',
                    'default_decoder': dict(
                        dec_embed_dim=768,
                        dec_num_heads=12,
                        mlp_ratio=4,
                        dec_depth=24,
                        head_depth=5,
                    ),
                    'variants': {}
                },
                'large':{
                    'builder': dinov2_vitl14_reg,
                    'model_dir': 'ckpts/dinov2-large',
                    'default_decoder': dict(
                        dec_embed_dim=1024,
                        dec_num_heads=16,
                        mlp_ratio=4,
                        dec_depth=36,
                        head_depth=5,
                    ),
                    'variants':{
                        '0.95B': dict(
                            dec_embed_dim=1024,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=36,
                            head_depth=5,
                        ),
                        '1.6B': dict(
                            dec_embed_dim=1024,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=72,
                            head_depth=10,
                        )
                    }
                },
                'giant':{
                    'builder': dinov2_vitg14_reg,
                    'model_dir': 'ckpts/dinov2-giant',
                    'default_decoder': dict(
                        dec_embed_dim=1536,
                        dec_num_heads=16,
                        mlp_ratio=4,
                        dec_depth=72,
                        head_depth=10,
                    ),
                    'variants':{
                        '3.57B': dict(
                            dec_embed_dim=1536,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=72,
                            head_depth=10,
                        )
                    }
                }
            },

            'dinov3':{
                'small':{
                    'builder': dinov3_vits16,
                    'model_dir': 'ckpts/dinov3_vits16',
                    'default_decoder': None,
                    'variants': {},
                },
                'small+':{
                    'builder': dinov3_vits16plus,
                    'model_dir': 'ckpts/dinov3_vits16plus',
                    'default_decoder': None,
                    'variants':{}
                },
                'base':{
                    'builder': dinov3_vitb16,
                    'model_dir': 'ckpts/dinov3_vitb16',
                    'default_decoder': None,
                    'variants':{}
                },
                'large':{
                    'builder': dinov3_vitl16,
                    'model_dir': 'ckpts/dinov3_vitl16',
                    'default_decoder': None,
                    'variants':{}
                },
                'huge+':{
                    'builder': dinov3_vith16plus,
                    'model_dir': 'ckpts/dinov3_vith16plus',
                    'default_decoder': None,
                    'variants': {
                        '3Bv2': dict(
                            dec_embed_dim=2048,
                            dec_num_heads=32,
                            mlp_ratio=4,
                            dec_depth=36,
                            head_depth=5,
                        ),
                        '3B': dict(
                            dec_embed_dim=1792,
                            dec_num_heads=28,
                            mlp_ratio=4,
                            dec_depth=56,
                            head_depth=5,
                        ),
                        '5Bv3': dict(
                            dec_embed_dim=3072,
                            dec_num_heads=48,
                            mlp_ratio=4,
                            dec_depth=36,
                            head_depth=6,
                        ),
                        '5Bv2': dict(
                            dec_embed_dim=2560,
                            dec_num_heads=32,
                            mlp_ratio=4,
                            dec_depth=56,
                            head_depth=6,
                        ), 
                        '2.81B': dict(
                            dec_embed_dim=1280,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=80,
                            head_depth=10,
                        ),
                        '3.2B': dict(
                            dec_embed_dim=1280,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=100,
                            head_depth=10,
                        ),
                        '4.24B': dict(
                            dec_embed_dim=2560,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=40,
                            head_depth=6,
                        ),
                        '3.51B': dict(
                            dec_embed_dim=2048,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=48,
                            head_depth=6,
                        ),
                        '5.17B': dict(
                            dec_embed_dim=3072,
                            dec_num_heads=16,
                            mlp_ratio=4,
                            dec_depth=36,
                            head_depth=6,
                        ),
                        
                    }
                }
            }

        }
    

    def decode(self, hidden, N, H, W):
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

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std
        enc_dtype = self.encoder.patch_embed.proj.weight.dtype
        imgs = imgs.to(dtype=enc_dtype)

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # encode by dinov
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden = self.project_e2d(hidden)
        hidden = self.project_norm(hidden)

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
