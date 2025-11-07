"""
直接更新GS经过activation之后的, 不更新gs featire tensor
10.24: 增加了delta_GS 的 activation
10.24: bug修复，之前的update module的初始化有问题, 最后一层没有初始化成0，修复这个bug。 
总结： V3和V2的区别： 增加了delta_GS 的 activation
V4: 增加了 GS voxel feature 输入
V5: 使用PTV3 block 替代 KNN attention 
V7: 对V5进行降显存
"""
import os
from typing import *
import time
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
# from ...utils.random_utils import hammersley_sequence
from ...models.GSdecoder.base import SparseTransformerBase, SparseTransformerCrossBase, VoxelDensityMLP, TransformerBase, TransformerCrossBase
from ...representations import Gaussian
from ...ptv3 import PointTransformerV3_2

# from flash_attn import flash_attn_func

DEBUG=False
if DEBUG:
    out_dir = "outputs_TRELLISdev/test_results/test_differentiable_projection"
    os.makedirs(out_dir, exist_ok=True)

def quaternion_to_axis_angle(q):
    # 假设四元数 q 是形状为 (4,) 的张量，表示 [w, x, y, z]
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # 计算旋转角度
    angle = 2 * torch.acos(w)  # 旋转角度 = 2 * arccos(w)
    
    # 计算旋转轴（如果角度接近零，则轴可以是任意方向，因此设置为 0）
    sin_half_angle = torch.sqrt(1 - w * w)  # sin(θ / 2)
    
    if sin_half_angle < 1e-8:  # 处理非常小的旋转角度，避免除以零
        axis = torch.tensor([1.0, 0.0, 0.0])  # 默认旋转轴
    else:
        axis = torch.tensor([x, y, z]) / sin_half_angle  # 归一化虚部
    
    return axis, angle

def clean_mem():
    gc.collect()
    torch.cuda.empty_cache()

                

class KNNAttention(nn.Module):
    def __init__(self, 
                 in_channels=1024,
                 k=16,
                 model_channels=1024,
                 num_blocks_knn_attn=4,
                 num_heads=4,
                 num_head_channels=256,
                 mlp_ratio=4,
                 attn_mode='full',
                 window_size=1024,
                 pe_mode='ptv3pe',
                 use_fp16=False,
                 use_bf16=True,
                 use_checkpoint=True,
                 qk_rms_norm=False,
                 dropout=0.0):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.model_channel = model_channels
        self.attn = TransformerBase(
                                    in_channels=in_channels,
                                    model_channels=model_channels,
                                    num_blocks=num_blocks_knn_attn,
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
        
        self.initialize_weights()
        for module in [self.attn]:
            if use_fp16:
                module.convert_to_fp16()
            if use_bf16:
                module.convert_to_bf16()
        
        
    def initialize_weights(self) -> None:
        for module in [self.attn]:
            module.initialize_weights()
    
    @staticmethod
    def optimized_cdist_batch(A, k, batch_size=1024):
        N, _ = A.shape  # A 和 B 的形状是 [N, 3]
        knn_idx = torch.zeros(N, k, dtype=torch.long, device=A.device)  # 存储结果

        # 计算每个点的范数（即自内积）
        A_norm = (A * A).sum(dim=-1)  # [N]

        # 分块计算距离
        num_batches = (N + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            
            A_batch = A[start:end]  # 当前批次的 B 点
            A_batch_norm = (A_batch * A_batch).sum(dim=-1)  # [batch_size]            

            # 计算内积矩阵 A·B^T
            inner_product = torch.matmul(A_batch, A.T)  # [batch_size, N]

            # 计算距离矩阵: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A·B
            dist_matrix = A_norm.unsqueeze(0) + A_batch_norm.unsqueeze(1) - 2 * inner_product            

            # 获取每个 B_batch 中的点距离 A 中最近的 k 个点
            knn_idx[start:end] = dist_matrix.topk(k, dim=-1, largest=False)[1]

        return knn_idx
    
    def forward(self, x, pos, batch_size_cdist=2048, batch_size_attn=1024):
        # x: [B, M, C]  pos: [M, 3]
        with torch.no_grad():
            # 使用cdist 太慢
            # dist = torch.cdist(pos, pos)  # [B, M, M]
            # 使用 内积
            # pos_norm = torch.norm(pos, p=2, dim=-1) ** 2  # [B, M]
            # dist = pos_norm.unsqueeze(2) + pos_norm.unsqueeze(1) - 2 * torch.bmm(pos, pos.transpose(1, 2))
            # knn_idx = dist.topk(self.k, dim=-1, largest=False)[1]  # [B, M, k]
            knn_idx = self.optimized_cdist_batch(pos, self.k, batch_size=batch_size_cdist)
            knn_idx = knn_idx.unsqueeze(0)

            # gather 局部邻域特征
            B, M, C = x.shape
            neighbors = torch.gather(
                x.unsqueeze(2).expand(B, M, self.k, C),
                1,
                knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
            ).reshape(B * M, self.k, C)  # [B*M, k, C]
            
            neighbor_pos = torch.gather(
                pos.unsqueeze(0).unsqueeze(2).expand(B, M, self.k, 3),  # 将 pos 扩展为 [B, M, k, 3]
                1,  # 在第 1 维度（N 维）上进行选择
                knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # 扩展 knn_idx 为 [B, N, k, 3]，每个邻居有 3 个坐标
            ).reshape(B * M, self.k, 3) # [B*M, k, 3]
            
        clean_mem()
        
        num_batches = (M + batch_size_attn - 1) // batch_size_attn
        att_out = torch.empty(0, self.model_channel).to(x.device)  # 假设每个张量的形状是 [3, 3]
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size_attn
            end = min((batch_idx + 1) * batch_size_attn, M)
            neighbors_batch = neighbors[start:end,...]  # 当前批次的 B 点
            neighbor_pos_batch = neighbor_pos[start:end,...]
            
            att_out_batch = self.attn(neighbors_batch, neighbor_pos_batch)
            # 对邻域进行聚合：按 k 维度加权平均（可以选择求和或者加权）
            att_out_batch = att_out_batch.mean(dim=1)  # [B*M, C]，对每个点的 k 邻居求平均
            att_out = torch.cat((att_out, att_out_batch), dim=0)
        # reshape 回原来的形状: [B, M, C]
        out = att_out.view(B, M, self.model_channel)  # [B, M, C]
        return out

# ------------------------------
# 更新模块 MLP
# ------------------------------
class UpdateModule(nn.Module):
    def __init__(self, dim, g_enc_dim, g_dim, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.g_enc_dim = g_enc_dim
        self.g_dim = g_dim
        self.mlp_dg = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            # nn.Linear(2*dim, 2*dim),
            # nn.ReLU(),
            # nn.Linear(2*dim, dim),
            # nn.ReLU(),
            nn.Linear(dim, g_dim)
        )
    
    def initialize_weights(self):
        for i, layer in enumerate(self.mlp_dg):
            if isinstance(layer, (nn.Linear, sp.SparseLinear)):
                if i == len(self.mlp_dg) - 1:
                    # 对最后一层使用 Zero 初始化
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:
                    # 对其他层使用 Kaiming 初始化
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _forward(self, x):
        return self.mlp_dg(x).squeeze()  # 默认前向计算

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
        
class EncGSModule(nn.Module):
    def __init__(self, g_dim, g_enc_dim, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.g_enc_dim = g_enc_dim
        self.g_dim = g_dim
        self.enc_gs = nn.Sequential(
                nn.Linear(g_dim, g_enc_dim//2),
                nn.ReLU(),
                nn.Linear(g_enc_dim//2, g_enc_dim)
            )
    
    def initialize_weights(self):
        for i, layer in enumerate(self.enc_gs):
            if isinstance(layer, (nn.Linear, sp.SparseLinear)):
                # if i == len(self.mlp_dg) - 1:
                #     # 对最后一层使用 Zero 初始化
                #     nn.init.zeros_(layer.weight)
                #     if layer.bias is not None:
                #         nn.init.zeros_(layer.bias)
                # else:
                # 对其他层使用 Kaiming 初始化
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _forward(self, x):
        return self.enc_gs(x)  # 默认前向计算

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

# ------------------------------
# 主模块：Recurrent Gaussian Update
# ------------------------------
class RefinerV7(nn.Module):
    def __init__(self, 
                 g_dim=64,
                 g_enc_dim=128, 
                 z_dim=1024,
                 e_dim=192,
                 k=16,
                 model_channels_e=256,
                 model_channels_knn=512,
                 num_blocks_knn_attn=4,
                 num_blocks_glb_attn=1,
                 num_heads=4,
                 num_head_channels=256,
                 mlp_ratio=4,
                 attn_mode='full',
                 window_size=1024,
                 pe_mode='ptv3pe',
                 use_fp16=False,
                 use_bf16=True,
                 use_checkpoint=True,
                 qk_rms_norm=False,
                 max_delta=None,
                 representation_config=None,
                 dropout=0.0):
        super().__init__()
        self.rep_config = representation_config 
        self.sh_degree = self.rep_config['sh_degree'] if 'sh_degree' in self.rep_config else 0
        self.num_features = 3 * ((self.sh_degree+1)**2)
        
        if g_dim % num_heads != 0 :
            self.enc_gs = EncGSModule(g_dim, g_enc_dim, use_checkpoint=use_checkpoint)
            # self.enc_gs = nn.Sequential(
            #     nn.Linear(g_dim, g_enc_dim//2),
            #     nn.ReLU(),
            #     nn.Linear(g_enc_dim//2, g_enc_dim)
            # )
        else:
            self.enc_gs = None
            g_enc_dim = g_dim
            
        assert z_dim % num_heads == 0, f"z_dim ({z_dim}) must be divisible by num_heads ({num_heads})"    
        assert e_dim % num_heads == 0, f"e_dim ({e_dim}) must be divisible by num_heads ({num_heads})"
                
        self.global_attention = TransformerBase(
                                                in_channels=e_dim,
                                                model_channels=model_channels_e,
                                                num_blocks=num_blocks_glb_attn,
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
        if use_fp16:
            attn_dtype = 'fp16'
        elif use_bf16:  
            attn_dtype = 'bf16'
        else:
            attn_dtype = 'fp32'
        
        
        # self.knnattn = PointTransformerV3_2(
        #                             in_channels=g_enc_dim + z_dim + model_channels_e,
        #                             # in_channels = 3, # ['coord]
        #                             stride=(2,),
        #                             enc_depths=(1,1,),
        #                             enc_channels=(512,1024),
        #                             enc_num_head=(32,64),
        #                             enc_patch_size=(1024,1024,),
        #                             dec_depths=(1,),
        #                             dec_channels=(512),
        #                             dec_num_head=(32),
        #                             dec_patch_size=(1024,),
        #                             enable_cpe_block=False,
        #                             enable_flash=True,
        #                             attn_dtype=attn_dtype,
        #                             enable_ca_flash=False,
        #                             drop_path=0.3,
        #                             shuffle_orders=True,
        #                             cls_mode=False, #True for enc only; False for enc-dec backbone        
        #                         )
        
        self.knnattn = PointTransformerV3_2(
                                            in_channels=g_enc_dim + z_dim + model_channels_e,
                                            # in_channels = 3, # ['coord]
                                            stride=(),
                                            enc_depths=(num_blocks_knn_attn,),
                                            enc_channels=(model_channels_knn,),
                                            enc_num_head=(num_heads,),
                                            enc_patch_size=(1024,),
                                            enable_cpe_block=False,
                                            enable_flash=True,
                                            use_checkpoint=use_checkpoint,
                                            attn_dtype=attn_dtype,
                                            enable_ca_flash=False,
                                            drop_path=0.1,
                                            shuffle_orders=True,
                                            cls_mode=True, #True for enc only; False for enc-dec backbone        
                                        )
        
        self.update_module = UpdateModule(model_channels_knn,
                                          g_enc_dim=g_enc_dim, 
                                          g_dim=g_dim,
                                          use_checkpoint=use_checkpoint)
        
        # init weights and convert to bf16/fp16
        self.initialize_weights()
        
        for module in [self.global_attention]:
            if use_fp16:
                module.convert_to_fp16()
            if use_bf16:
                module.convert_to_bf16()
        
        if use_fp16:
            self.dtype = torch.float16
        elif use_bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float

        self.max_delta_xyz = max_delta['max_delta_xyz'] # 6/128
        self.max_delta_opa = max_delta['max_delta_opa']  # 0.1
        self.max_delta_rgb = max_delta['max_delta_rgb']  # 0.1
        self.max_delta_sca = max_delta['max_delta_sca']  # 0.05
                
    def initialize_weights(self) -> None:
        for module in [self.global_attention]:
            module.initialize_weights()
        
        # self-define  output layers
        out_layers = [self.update_module]
        for layer in out_layers:
            layer.initialize_weights()
    
    
    def to_representation(self, infeats: List[torch.Tensor]) -> Gaussian: # aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]
        """
        Convert a batch of network outputs to 3D gaussians.

        Args:
            gaussians : Gaussain class for activation, save, load, etc.
            infeats: list of gaussians. element in list is gaussians in different level. shape: [N, 14]
        Returns:
            list of final gaussians
        """
        gauss = Gaussian(
            sh_degree=self.sh_degree,
            mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
            scaling_bias = self.rep_config['scaling_bias'],
            opacity_bias = self.rep_config['opacity_bias'],
            scaling_activation = self.rep_config['scaling_activation']
        )
        
        keys = ['_xyz', '_opacity', '_scaling', '_rotation',  '_features_dc', '_featires_rest' ]
        nums = [3, 1, 3, 4, 3, self.num_features-3]

        start_dim = 0
        for dim, key in zip(nums, keys):
            if key == '_features_dc':
                setattr(gauss, key, infeats[:, start_dim:start_dim+dim].unsqueeze(1))
            else:
                setattr(gauss, key, infeats[:, start_dim:start_dim+dim])
            start_dim += dim
        
        return gauss


    def forward(self, old_gs, z, E, batch_size_cdist=4096, batch_size_attn=4096):
        """
        old_gs: [B, M, 14] — Gaussian attributes
        z: [B, M, 1024] — Gaussian attributes
        E: [B, N, e_dim] — Rendering error tokens
        """        
        # with torch.cuda.amp.autocast(dtype=self.dtype):
        if self.enc_gs:
            g_enc = self.enc_gs(old_gs)
        else:
            g_enc = old_gs
    
        # Step 1: Global attention over rendering errors
        e_global = self.global_attention(E.unsqueeze(0), old_gs[..., :3])  # [N, model_chn]

        clean_mem()
        
        # Step 2: Concatenate Gaussian, position, and error info
        x = torch.cat([g_enc, z, e_global], dim=-1)  # [B, M, g_dim+z_dim+model_chn]

        # Step 3: Recurrent local aggregation (4 kNN attention blocks)
        point = dict(
            coord=old_gs[..., :3],
            feat=x,
            offset=torch.tensor([x.shape[0]]).to(x.device),
            grid_size=1e-3
        )
        x = self.knnattn(point)
        x = x['feat']
        
        clean_mem()
        
        # Step 4: MLP update
        dg = self.update_module(x)
        
        clean_mem()
            
        dg_xyz = dg[...,  :3]
        dg_opa = dg[..., 3:4]
        dg_sca = dg[..., 4:7]
        dg_rot = dg[..., 7:11]
        dg_fdc = dg[..., 11:]
        
        dg_xyz = self.max_delta_xyz * torch.tanh(dg_xyz)
        dg_opa = torch.sigmoid(dg_opa) * self.max_delta_opa
        dg_sca = torch.clamp(dg_sca, max=self.max_delta_sca)
        dg_fdc = torch.sigmoid(dg_fdc) * self.max_delta_rgb      
        
        dg = torch.cat([
            dg_xyz, dg_opa, dg_sca, dg_rot, dg_fdc], dim=-1)

        
        g_next = old_gs + dg
        g_next_normalized = F.normalize(g_next[..., 7:11])
        g_next = g_next.clone()
        g_next[..., 7:11] = g_next_normalized
        
        clean_mem()
        
        return g_next

# ------------------------------
# 测试
# ------------------------------
if __name__ == "__main__":
    B, M, N = 1, 2048, 512
    g_dim, z_dim, e_dim = 14, 16, 128
    # 实际上 g_dim = 14, e_dim = C, z_dim = 1024
    # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    model = Refiner(g_dim=g_dim, g_enc_dim=128, z_dim=z_dim, e_dim=e_dim).cuda()
    
    g = torch.randn(B, M, g_dim, device="cuda")
    z = torch.randn(B, M, z_dim, device="cuda")
    E = torch.randn(B, N, e_dim, device="cuda")

    g_next, z_next, dg, dz = model(g, z, E)
    print(f"✅ g_next: {g_next.shape}, z_next: {z_next.shape}")

