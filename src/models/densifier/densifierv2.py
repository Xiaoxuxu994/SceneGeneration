from typing import *
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from ...models.structured_latent_vae.base import SparseTransformerBase, SparseTransformerCrossBase, VoxelDensityMLP, TransformerBase, TransformerCrossBase
from ...representations import Gaussian, Gaussian2D

class DensifierV2(nn.Module):
    def __init__(
        self,
        # voxel_size: float,  # 固定的体素大小
        model_channels: int,
        voxel_channels: int = 1024,
        gswithgrad_channels: int = 14+1,
        num_blocks_GSenc: int = 1,
        num_blocks_VoxGSenc: int = 1,
        num_blocks_qenc: int = 1,
        num_blocks_MCA: int = 1,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope", "ptv3pe"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        num_hidden_dims: int = 4,
        init_ratio: float = 1.,
        voxel_mode: str = "center",
        dropout: float = 0.1, # << New: dropout rate
        # voxel_bound: Optional[tuple] = (-0.025, 0.025),  # 默认为[-0.5, 0.5]的网格范围   
    ):
        super().__init__()
        self.rep_config = representation_config        
        self.sh_degree = self.rep_config['sh_degree'] if 'sh_degree' in self.rep_config else 0
        self.num_features_dc = 3 * ((self.sh_degree+1)**2)
        self.num_gs = self.rep_config['num_gaussians']
        self.voxel_size = self.rep_config['voxel_size']

        # encoder for feat_voxel can be large
        self.VoxAndGs_enc = TransformerBase(
            in_channels=voxel_channels+model_channels,
            model_channels=model_channels,
            num_blocks=num_blocks_VoxGSenc,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dropout=dropout, # << New: dropout rate
        )
        
        # encoder for GS_attribute
        self.Gs_enc = TransformerBase(
            in_channels=gswithgrad_channels,
            model_channels=model_channels,
            num_blocks=num_blocks_GSenc,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dropout=dropout, # << New: dropout rate
        )
        
        # encoder for query can be large
        self.q_enc = TransformerBase(
            in_channels=3,
            model_channels=model_channels,
            num_blocks=num_blocks_qenc,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dropout=dropout, # << New: dropout rate
        )
        
        self.cross_attn = TransformerCrossBase(
                in_channels=model_channels,
                model_channels=model_channels,
                ctx_channels=model_channels,
                num_blocks=num_blocks_MCA,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                mlp_ratio=mlp_ratio,
                attn_mode='full',
                window_size=1024,
                pe_mode=pe_mode,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                use_checkpoint=use_checkpoint,
                qk_rms_norm=qk_rms_norm,    
                dropout=dropout,        
            )
        
        def _build_outlayer(model_channels, out_channels, num_hidden_dims=4, mode='sp'):
            hidden_dims = np.linspace(model_channels, out_channels, num_hidden_dims+2).astype(np.int16)[1:-1].tolist()
            layers = []
            prev_dim = model_channels
            if mode == 'sp':
                for dim in hidden_dims:
                    layers.append(sp.SparseLinear(prev_dim, dim))
                    layers.append(sp.SparseReLU())  # 激活函数
                    prev_dim = dim
                layers.append(sp.SparseLinear(prev_dim, out_channels))
            elif mode == 'nn':
                for dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, dim))
                    layers.append(nn.ReLU())  # 激活函数
                    prev_dim = dim
                layers.append(nn.Linear(prev_dim, out_channels))
            out_layer = nn.Sequential(*layers)
            return out_layer
        
        self.num_gs = 1
        self.out_layer_xyz = _build_outlayer(model_channels, self.num_gs * 3, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_fdc = _build_outlayer(model_channels, self.num_gs * self.num_features_dc, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_sca = _build_outlayer(model_channels, self.num_gs * 3, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_rot = _build_outlayer(model_channels, self.num_gs * 4, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_opa = _build_outlayer(model_channels, self.num_gs, num_hidden_dims=num_hidden_dims, mode='nn')
        
        # init weights and convert to bf16/fp16
        self.initialize_weights()
            
        for module in [self.q_enc, self.VoxAndGs_enc, self.Gs_enc, self.cross_attn]:
            if use_fp16:
                module.convert_to_fp16()
            if use_bf16:
                module.convert_to_bf16()
    
    def initialize_weights(self) -> None:
        for module in [self.q_enc, self.VoxAndGs_enc, self.Gs_enc, self.cross_attn]:
            module.initialize_weights()
            
        # # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        # nn.init.constant_(self.out_layer.bias, 0)
        
        # self-define  output layers
        out_layers = [
            self.out_layer_xyz, self.out_layer_fdc, self.out_layer_sca, self.out_layer_rot, self.out_layer_opa,
            ]
        for i, layer in enumerate(out_layers):
            if isinstance(layer, (nn.Linear, sp.SparseLinear)):
                if i == len(layer) - 1:
                    # 对最后一层使用 Zero 初始化
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:
                    # 对其他层使用 Kaiming 初始化
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def to_representation(self, gaussians: Gaussian, infeats: List[torch.Tensor]) -> Gaussian: # aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]
        """
        Convert a batch of network outputs to 3D gaussians.

        Args:
            gaussians : Gaussain class for activation, save, load, etc.
            infeats: list of gaussians. element in list is gaussians in different level. shape: [N, 14]
        Returns:
            list of final gaussians
        """
        keys = ['_xyz', '_opacity', '_scaling', '_rotation',  '_features_dc', ]
        nums = [3, 1, 3, 4, self.num_features_dc,]
        # infeats = torch.cat(infeats, dim=0)
        start_dim = 0
        for dim, key in zip(nums, keys):
            if key == '_features_dc':
                setattr(gaussians, key, infeats[:, start_dim:start_dim+dim].unsqueeze(1))
            else:
                setattr(gaussians, key, infeats[:, start_dim:start_dim+dim])
            start_dim += dim
            
        return gaussians
    
    def forward(self, feat_voxel: torch.Tensor, attr_GS: torch.Tensor, query: torch.Tensor = None,
                voxel_align_ratio=5.):   
        """
        x: 上一层得到的GS feat + GS grad [N, 14+14]
        query: 想要增加的GS query [M, 14]
        """
        Gaussians = Gaussian(
            sh_degree=self.sh_degree,
            mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
            scaling_bias = self.rep_config['scaling_bias'],
            opacity_bias = self.rep_config['opacity_bias'],
            scaling_activation = self.rep_config['scaling_activation']
        )
        Gaussians.from_xyz(attr_GS[:,:3])
        Gaussians.from_opacity(attr_GS[:, 3:4])
        Gaussians.from_scaling(attr_GS[:, 4:7])
        Gaussians.from_rotation(attr_GS[:, 7:11])
        Gaussians.from_features(attr_GS[:, 11:14])
        
        feat_GS = self.Gs_enc.forward(attr_GS.unsqueeze(0), coords=attr_GS[:,:3]) # 
        feat_GS = feat_GS.type(feat_GS.dtype)
        feat_GS = F.layer_norm(feat_GS, feat_GS.shape[-1:]) 
        
        num_voxel, dim_voxel = feat_voxel.shape
        feat_voxel = feat_voxel.unsqueeze(1).repeat(1, 4, 1)  # 先加一个维度，然后沿着这个维度重复4次
        feat_voxel = feat_voxel.view(-1, dim_voxel)
        x = torch.cat([feat_voxel, feat_GS], dim=-1)
        
        q_anchors = query.clone()
        
        kv = self.VoxAndGs_enc.forward(x.unsqueeze(0), coords=attr_GS[:,:3]) # 
        kv = kv.type(x.dtype)
        kv = F.layer_norm(kv, kv.shape[-1:])
        
        q = self.q_enc.forward(query.unsqueeze(0), coords=q_anchors) # 
        q = q.type(x.dtype)
        q = F.layer_norm(q, q.shape[-1:])
        
        
        h = self.cross_attn(q.unsqueeze(0), kv.unsqueeze(0), coords=q_anchors, coords_cxt=attr_GS[:,:3])    
        h = h.type(x.dtype)    
        h = F.layer_norm(h, h.shape[-1:])

        # h = self.out_layer(h) # h: [M, 14]
        delta_xyz = self.out_layer_xyz(h)
        delta_xyz = voxel_align_ratio * 0.5 * self.voxel_size * torch.tanh(delta_xyz)
        feats_xyz = delta_xyz + q_anchors
        feats_fdc = self.out_layer_fdc(h)
        feats_sca = self.out_layer_sca(h)
        feats_rot = self.out_layer_rot(h)
        feats_opa = self.out_layer_opa(h)
        new_feats = torch.cat([feats_xyz, feats_opa, feats_sca, feats_rot, feats_fdc], dim=-1)  # [M, 14]
        
        old_feats_xyz = Gaussians._xyz
        old_feats_opa = Gaussians._opacity
        old_feats_sca = Gaussians._scaling
        old_feats_rot = Gaussians._rotation
        old_feats_fdc = Gaussians._features_dc
        old_feats = torch.cat([old_feats_xyz, old_feats_opa, old_feats_sca, old_feats_rot, old_feats_fdc], dim=-1)  # [N, 14]
        
        all_feats = torch.cat([old_feats, new_feats], dim=0)
        # all_gs.append(new_feats)
        return [self.to_representation(Gaussians, all_feats)] # list [ gs_feats Tensor Nx14]

