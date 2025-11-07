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

class DensifierSparse(nn.Module):
    def __init__(
        self,
        # voxel_size: float,  # 固定的体素大小
        model_channels: int,
        latent_channels: int,
        num_blocks_kvenc: int = 1,
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

        # encoder for query can be large
        self.kv_enc = SparseTransformerBase(
            in_channels=1024+14+14,
            model_channels=model_channels,
            num_blocks=num_blocks_kvenc,
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
            isPTV3PELog=pe_mode=='ptv3pe',
        )
        
        # encoder for query can be large
        self.q_enc = SparseTransformerBase(
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
            isPTV3PELog=pe_mode=='ptv3pe',
        )
        
        self.cross_attn = SparseTransformerCrossBase(
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
                isPTV3PELog=pe_mode=='ptv3pe',
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
        mode = 'nn'
        self.out_layer_xyz = _build_outlayer(model_channels, self.num_gs * 3, num_hidden_dims=num_hidden_dims, mode=mode)
        self.out_layer_fdc = _build_outlayer(model_channels, self.num_gs * self.num_features_dc, num_hidden_dims=num_hidden_dims, mode=mode)
        self.out_layer_sca = _build_outlayer(model_channels, self.num_gs * 3, num_hidden_dims=num_hidden_dims, mode=mode)
        self.out_layer_rot = _build_outlayer(model_channels, self.num_gs * 4, num_hidden_dims=num_hidden_dims, mode=mode)
        self.out_layer_opa = _build_outlayer(model_channels, self.num_gs, num_hidden_dims=num_hidden_dims, mode=mode)
        
        # init weights and convert to bf16/fp16
        self.initialize_weights()
            
        for module in [self.kv_enc, self.q_enc, self.cross_attn]:
            if use_fp16:
                module.convert_to_fp16()
            if use_bf16:
                module.convert_to_bf16()
    
    def initialize_weights(self) -> None:
        for module in [self.kv_enc, self.q_enc, self.cross_attn]:
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
    
    @staticmethod
    def convert_xyz_to_coords(coords: torch.Tensor, scale: float = 10000.0) -> torch.Tensor:
        """
        浮点坐标一对一映射到整数坐标
        scale 控制精度，比如 1000 表示保留 3 位小数
        """
        return (coords * scale).round().long()
    
    
    def forward(self, feat_voxel: torch.Tensor, feat_GS: torch.Tensor, query: torch.Tensor = None,
                voxel_align_ratio=1.):   
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
        Gaussians.from_xyz(feat_GS[:,:3])
        Gaussians.from_opacity(feat_GS[:, 3:4])
        Gaussians.from_scaling(feat_GS[:, 4:7])
        Gaussians.from_rotation(feat_GS[:, 7:11])
        Gaussians.from_features(feat_GS[:, 11:14])
        
        num_voxel, dim_voxel = feat_voxel.shape
        feat_voxel = feat_voxel.unsqueeze(1).repeat(1, 4, 1)  # 先加一个维度，然后沿着这个维度重复4次
        feat_voxel = feat_voxel.view(-1, dim_voxel)
        x = torch.cat([feat_voxel, feat_GS], dim=-1)
 
        # all_gs = [feat_GS[:,:14]]
        
        voxel_coords = self.convert_xyz_to_coords(feat_GS[:,:3], scale=1e4)  # [N, 3] # 把anchor position放大并转成整数坐标
        x = sp.SparseTensor(feats=x, 
                            coords=torch.cat([
                                torch.zeros(x.shape[0], 1).int().to(feat_voxel.device),
                                voxel_coords.int().to(feat_voxel.device),
                            ], dim=1))  # [N, C]
        # all_gs = [feat_GS[:,:14]]
        
        q_anchors = self.convert_xyz_to_coords(query, scale=1e5)  # [M, 3] # query position放大并转成整数坐标
        q = sp.SparseTensor(feats=query, 
                            coords=torch.cat([
                                torch.zeros(query.shape[0], 1).int().to(feat_voxel.device), 
                                q_anchors.int().to(feat_voxel.device)
                            ], dim=1))  # type: ignore # [M, 3]
        
        kv = self.kv_enc.forward(x) # 
        kv = kv.type(x.dtype)
        kv = kv.replace(F.layer_norm(kv.feats, kv.feats.shape[-1:])) # type: ignore
        
        q = self.q_enc.forward(q) # 
        q = q.type(x.dtype)
        q = q.replace(F.layer_norm(q.feats, q.feats.shape[-1:]))

        h = self.cross_attn(q, kv)    
        h = h.type(x.dtype)    
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))

        # h = self.out_layer(h) # h: [M, 14]
        delta_xyz = self.out_layer_xyz(h.feats)
        delta_xyz = voxel_align_ratio * 0.5 * self.voxel_size * torch.tanh(delta_xyz)
        feats_xyz = delta_xyz + query
        feats_fdc = self.out_layer_fdc(h.feats)
        feats_sca = self.out_layer_sca(h.feats)
        feats_rot = self.out_layer_rot(h.feats)
        feats_opa = self.out_layer_opa(h.feats)
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

