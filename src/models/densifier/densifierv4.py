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

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class LightFeatureUNet(nn.Module):
    def __init__(self, in_chns=1, out_chns=1024, size=(168, 252)):
        super().__init__()

        # Downsampling (只下采样到 1/3 左右，不到 1/4)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_chns, 64, kernel_size=3, stride=2, padding=1),   # 588x882 -> 294x441
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),       # 294x441 -> 147x221
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),      # 保持 147x221
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Upsampling path with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),  # 保持 147x221
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(size=size, mode='bilinear', align_corners=False),
            nn.Conv2d(256+256, out_chns, kernel_size=1)  # 融合 skip connection
        )

    def forward(self, x):
        # Down path
        x1 = self.down1(x)   # 294x441
        x2 = self.down2(x1)  # 147x221
        x3 = self.down3(x2)  # 147x221

        # Bottleneck
        xb = self.bottleneck(x3)  # 147x221

        # Up path with skip connections
        xu1 = self.up1(xb)        # 147x221
        xu1 = torch.cat([xu1, x3], dim=1)  # skip connection

        out = self.up2(xu1)       # 168x252

        return out

class DensifierV4(nn.Module):
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
        enc_infos=["camRay",], #  "camRay" "depth"
        dropout: float = 0.1, # << New: dropout rate
        # voxel_bound: Optional[tuple] = (-0.025, 0.025),  # 默认为[-0.5, 0.5]的网格范围   
    ):
        super().__init__()
        
        self.enc_infos = enc_infos
        in_chns = 0
        if "rgbOri" in self.enc_infos:
            in_chns += 3
        if "depth" in self.enc_infos:
            in_chns += 1
        if "camRay" in self.enc_infos:
            in_chns += 6
        if "normal" in self.enc_infos:
            in_chns += 3
        self.featSize =  (168, 252) # (42, 63) #
        self.inputSize = (1168, 1752) # (2352//4, 3528//4) # 原始输入DINO图像大小//4
        self.featEnc = LightFeatureUNet(in_chns=in_chns, out_chns=model_channels, size=self.featSize)
        self.featEnc.apply(init_weights)
        
        # for PROJECTOR
        self.depth_min = 1e-6
        self.depth_max = 4.0
        self.acc = 4 / 128
        
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
   
    # @staticmethod
    # def optimized_cdist(A, B, k):
    #     N, _ = A.shape  # A 和 B 的形状是 [N, 3]

    #     # 计算每个点的范数（即自内积）
    #     A_norm = (A * A).sum(dim=-1)  # [N]
    #     B_norm = (B * B).sum(dim=-1)  # [N]

    #     # 计算内积矩阵 A·B^T
    #     inner_product = torch.matmul(A, B.T)  # [N, N]

    #     # 计算距离矩阵: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A·B
    #     dist_matrix = A_norm.unsqueeze(1) + B_norm.unsqueeze(0) - 2 * inner_product  # [N, N]

    #     # 获取每个 B 中的点距离 A 中最近的 k 个点
    #     knn_idx = dist_matrix.topk(k, dim=-1, largest=False)[1]  # [N, k]

    #     return knn_idx
    
    @staticmethod
    def optimized_cdist_batch(A, B, k, batch_size=1024):
        N, _ = A.shape  # A 和 B 的形状是 [N, 3]
        knn_idx = torch.zeros(N, k, dtype=torch.long, device=A.device)  # 存储结果

        # 计算每个点的范数（即自内积）
        A_norm = (A * A).sum(dim=-1)  # [N]
        B_norm = (B * B).sum(dim=-1)  # [N]

        # 分块计算距离
        num_batches = (N + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            
            B_batch = B[start:end]  # 当前批次的 B 点

            # 计算内积矩阵 A·B^T
            inner_product = torch.matmul(B_batch, A.T)  # [batch_size, N]

            # 计算距离矩阵: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A·B
            dist_matrix = A_norm[start:end].unsqueeze(1) + B_norm.unsqueeze(0)[start:end] - 2 * inner_product

            # 获取每个 B_batch 中的点距离 A 中最近的 k 个点
            knn_idx[start:end] = dist_matrix.topk(k, dim=-1, largest=False)[1]

        return knn_idx
    
    @staticmethod
    def knn_average(A, knn_idx):
        # knn_idx: [N, k]，表示B中每个点的k个最近邻的索引
        # A: [N, 3]，表示A中的N个点，假设A中每个点有3个维度
        N, _ = A.shape
        # 通过knn_idx获取A中每个点对应的k个最近邻
        knn_points = A[knn_idx]  # knn_points: [N, k, 3]，每个B[i]对应的A中的k个最近邻
        # 对每个B[i]，对A中的k个最近邻求均值
        knn_avg = knn_points.mean(dim=1)  # [N, 3]，对每个B[i]的K个邻居点求均值
        return knn_avg

    @staticmethod
    def normalize_c2ws(w2cs: torch.Tensor, T_norm: torch.Tensor, scale: float) -> torch.Tensor:
        """
        将原始 w2c 外参变换到归一化空间，输出归一化后的 c2w，并修正旋转部分为正交矩阵。

        Args:
            w2cs: Tensor of shape [B, 4, 4], 原始世界到相机的变换矩阵
            T_norm:   Tensor of shape [4, 4], 世界坐标归一化矩阵（用于归一化 mesh）

        Returns:
            c2w_norm: Tensor of shape [B, 4, 4], 归一化坐标下的相机位姿 c2w（旋转已单位化）
        """

        assert w2cs.ndim == 3 and w2cs.shape[1:] == (4, 4), "Expected ori_w2cs to be [B, 4, 4]"

        B = w2cs.shape[0]
        R_ori = w2cs[:, :3,:3]
        t_ori = w2cs[:, :3, 3]
        cpos = torch.bmm(-R_ori.transpose(1,2), t_ori.unsqueeze(-1))  # [B,3] ori camera position
        cpos_norm = cpos.squeeze(-1) / scale
        c2w_norm = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(R_ori.device)  # [B, 4, 4]
        c2w_norm[:, :3, :3] = R_ori.transpose(1, 2)  # 转置旋转矩阵
        c2w_norm[:, :3, 3] = cpos_norm
        return c2w_norm

    @staticmethod
    def get_ray_embedding_batch(input_Ks, input_c2ws, h, w, depths=None, opengl=True):
        """
        Args:
            input_Ks: [B, 3, 3]   (normalized! fx/w, fy/h, cx/w, cy/h)
            input_c2ws: [B, 4, 4]
            h, w: image height and width
            depths: [B, H, W] or None
            opengl: whether to flip y-axis (OpenGL convention)
        Returns:
            ray_embedding: [B, 6, H, W]
        """

        B = input_Ks.shape[0]

        # === 1. Unnormalize intrinsics ===
        K_pixel = input_Ks.clone()
        K_pixel[:, 0, :] *= w  # fx cx
        K_pixel[:, 1, :] *= h  # fy cy

        fx = K_pixel[:, 0, 0].view(B, 1, 1)  # [B, 1, 1]
        fy = K_pixel[:, 1, 1].view(B, 1, 1)  # [B, 1, 1]
        cx = K_pixel[:, 0, 2].view(B, 1, 1)  # [B, 1, 1]
        cy = K_pixel[:, 1, 2].view(B, 1, 1)  # [B, 1, 1]

        # === 2. Generate pixel grid [H, W] ===
        grid_x, grid_y = torch.meshgrid(
            torch.arange(w, device=input_Ks.device),
            torch.arange(h, device=input_Ks.device),
            indexing='xy'
        )  # [H, W]
        grid_x = grid_x + 0.5  # center of pixel
        grid_y = grid_y + 0.5

        # Expand to batch: [B, H, W]
        x = repeat(grid_x, 'h w -> b h w', b=B)
        y = repeat(grid_y, 'h w -> b h w', b=B)

        # === 3. Compute direction in camera space ===
        dirs_x = (x - cx) / fx
        dirs_y = (y - cy) / fy * (-1.0 if opengl else 1.0)
        dirs_z = torch.ones_like(dirs_x) * (-1.0 if opengl else 1.0)

        camera_dirs = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # [B, H, W, 3]

        # === 4. Transform to world space ===
        R = input_c2ws[:, :3, :3]  # [B, 3, 3]
        # R = -R.transpose(1, 2)
        # R[:, :, 2] *= -1  # 翻转 Z 轴
        rays_d = torch.matmul(camera_dirs.view(B, -1, 3), R.transpose(1, 2))  # [B, HW, 3]
        rays_d = rays_d.view(B, h, w, 3)    
        rays_d = F.normalize(rays_d, dim=-1)  # 加上这句，确保单位方向

        # === 5. Ray origins ===
        t = input_c2ws[:, :3, 3]  # [B, 3]
        # t[:, :2] *= -1  # 翻转 x/y
        rays_o = t.view(B, 1, 1, 3).expand(-1, h, w, -1)  # [B, H, W, 3]

        # === 6. Optional depth mask ===
        if depths is not None:
            mask = depths > 0  # [B, H, W]
            rays_o = rays_o * mask.unsqueeze(-1)
            rays_d = rays_d * mask.unsqueeze(-1)

        # === 7. Combine ray dir + cross(ray_o, ray_d) ===
        cross = torch.cross(rays_o, rays_d, dim=-1)  # [B, H, W, 3]
        ray_embedding = torch.cat([rays_d, cross], dim=-1)  # [B, H, W, 6]
        ray_embedding = ray_embedding.permute(0, 3, 1, 2).contiguous()  # [B, 6, H, W]
        
        return ray_embedding
    
    @staticmethod
    def batch_project_and_aggregate_chunkwise(points, features, depths, intrinsics, w2cs, 
                                              T_norm=None, depth_tol=0.05,
                                              fill_empty=True, knn_k=8, knn_eps=1e-6,
                                              chunk_size_V=4, chunk_size_N=10000):
        """
        分块计算版本, 避免显存OOM
        
        Args:
            points: [N, 3] 3D点世界坐标
            features: [V, C, H, W] 多视角特征图
            depths: [V, H, W] 多视角深度图（单位与相机坐标系一致）
            intrinsics: [V, 3, 3] 相机内参
            w2cs: [V, 4, 4] 世界->相机 外参
            T_norm: 归一化矩阵
            depth_tol: 深度threshold
            fill_empty: 是否对无特征点做KNN插值
            knn_k: KNN邻居数
            knn_eps: 防除零小数
            chunk_size_V: 每次处理的视角块大小
            chunk_size_N: 每次处理的点数块大小
        
        Returns:
            aggregated_feats: [N, C] 聚合后的点特征
            visibility_mask: [N] 是否至少在一个视角可见
        """
        device = points.device
        V, C, H, W = features.shape
        N = points.shape[0]

        aggregated_feats = torch.zeros((N, C), device=device)
        weight_sum = torch.zeros((N, 1), device=device)

        # 齐次坐标提前计算，避免重复
        homog_points = torch.cat([points, torch.ones((N, 1), device=device)], dim=-1)  # [N,4]

        visibility_mask_all = torch.zeros((N,), dtype=torch.bool, device=device)
        
        # 分块循环处理视角
        for v_start in range(0, V, chunk_size_V):
            v_end = min(v_start + chunk_size_V, V)

            features_chunk = features[v_start:v_end]    # [V_c,C,H,W]
            depths_chunk = depths[v_start:v_end]        # [V_c,H,W]
            intr_chunk = intrinsics[v_start:v_end]      # [V_c,3,3]
            w2c_chunk = w2cs[v_start:v_end]             # [V_c,4,4]

            V_c = v_end - v_start

            # 分块循环处理点
            for n_start in range(0, N, chunk_size_N):
                n_end = min(n_start + chunk_size_N, N)

                points_chunk = points[n_start:n_end]          # [N_c,3]
                homog_points_chunk = homog_points[n_start:n_end]  # [N_c,4]
                N_c = n_end - n_start

                # 世界坐标 -> 相机坐标 [V_c, N_c, 4]
                cam_points = torch.einsum('vij,nj->vni', w2c_chunk, homog_points_chunk)
                cam_points_xyz = cam_points[..., :3]  # [V_c,N_c,3]

                # 判断是否在相机前方
                z_cam = cam_points_xyz[..., 2]  # [V_c,N_c]
                if T_norm is not None:
                    z_cam = z_cam * T_norm[2, 2]
                in_front = z_cam > 1e-6

                # 归一化投影坐标
                z_safe = z_cam.clamp(min=1e-6)
                x_norm = cam_points_xyz[..., 0] / z_safe
                y_norm = cam_points_xyz[..., 1] / z_safe

                # 相机内参
                fx = intr_chunk[:, 0, 0].view(V_c, 1)  # [V_c,1]
                fy = intr_chunk[:, 1, 1].view(V_c, 1)
                cx = intr_chunk[:, 0, 2].view(V_c, 1)
                cy = intr_chunk[:, 1, 2].view(V_c, 1)

                # 投影像素坐标
                u = fx * x_norm + cx  # [V_c, N_c]
                v = fy * y_norm + cy  # [V_c, N_c]

                # 图像内范围判断
                valid_u = (u >= 0) & (u <= (W - 1))
                valid_v = (v >= 0) & (v <= (H - 1))
                in_img = valid_u & valid_v  # [V_c, N_c]

                # 归一化到[-1,1]，grid_sample要求
                u_norm = 2.0 * (u / (W - 1)) - 1.0  # [V_c, N_c]
                v_norm = 2.0 * (v / (H - 1)) - 1.0  # [V_c, N_c]
                grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2).to(depths.device)  # [V_c, N_c, 1, 2]
                grid = torch.clamp(grid, -1 + 1e-3, 1 - 1e-3)
                
                # 采样深度 [V_c, 1, H, W] 和grid采样 [V_c, N_c, 1, 2]
                sampled_depths = F.grid_sample(
                    depths_chunk,  # [V_c,1,H,W]
                    grid,                      # [V_c,N_c,1,2]
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1).squeeze(-1)  # [V_c, N_c]
                
                # 深度有效性判定
                depth_positive = sampled_depths > 0
                depth_close = torch.abs(sampled_depths - z_cam) < depth_tol
                depth_valid = depth_positive & depth_close  # [V_c, N_c]

                # 综合有效性mask
                valid_mask = in_front & in_img & depth_valid  # [V_c, N_c]
                
                any_visible = valid_mask.any(dim=0)  # [N_c]
                visibility_mask_all[n_start:n_end] |= any_visible

                # 从feature map采样特征
                sampled_feats = F.grid_sample(
                    features_chunk,          # [V_c, C, H, W]
                    grid,                   # [V_c, N_c, 1, 2]
                    mode='bilinear',
                    align_corners=True
                ).squeeze(-1)              # [V_c, C, N_c]
                sampled_feats = sampled_feats.permute(2, 0, 1)  # [N_c, V_c, C]
                
                # 加权平均（根据valid_mask）
                weights = valid_mask.float().permute(1, 0)  # [N_c, V_c]
                total_weights = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [N_c,1]
                weighted_feats = sampled_feats * weights.unsqueeze(-1)  # [N_c, V_c, C]
                feat_chunk = weighted_feats.sum(dim=1)  # [N_c, C]

                # 累加结果
                aggregated_feats[n_start:n_end] = aggregated_feats[n_start:n_end] + feat_chunk
                weight_sum[n_start:n_end] = weight_sum[n_start:n_end] + total_weights

        # 归一化
        aggregated_feats = aggregated_feats / weight_sum.clamp(min=1e-6)
        # visibility_mask = (weight_sum.squeeze(-1) > 0)
        visibility_mask = visibility_mask_all
        
        # 对无特征点做KNN插值补全
        if fill_empty:
            missing_mask = ~visibility_mask  # 无特征点mask
            if missing_mask.any():
                src_points = points[visibility_mask]  # [M,3]
                src_feats = aggregated_feats[visibility_mask]  # [M,C]
                tgt_points = points[missing_mask]  # [K,3]

                k = min(knn_k, src_points.shape[0])
                dists = torch.cdist(tgt_points, src_points, p=2)  # [K, M]
                knn_dists, knn_idx = torch.topk(dists, k, largest=False, dim=1)

                knn_weights = 1.0 / (knn_dists + knn_eps)
                knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # 归一化

                interpolated_feats = torch.sum(src_feats[knn_idx] * knn_weights.unsqueeze(-1), dim=1)  # [K,C]
    
                aggregated_feats[missing_mask] = interpolated_feats
                        
        torch.cuda.empty_cache()

        return aggregated_feats, visibility_mask
    
    def forward(self, 
                attr_GS: torch.Tensor, 
                query: torch.Tensor = None,
                rgbOris=None, depths=None, intrinsic=None,
                w2cs=None, scale=None, T_norm=None, normals=None,
                voxel_align_ratio=5.):   
        """
        增加了knn附近的特征聚合
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
        
        # 对GS本身的属性进行编码
        feat_GS = self.Gs_enc.forward(attr_GS.unsqueeze(0), coords=attr_GS[:,:3]) # 
        feat_GS = feat_GS.type(feat_GS.dtype)
        feat_GS = F.layer_norm(feat_GS, feat_GS.shape[-1:]) 
        
        ''' 通过图像特征反投影回GS位置得到GS feature '''
        # ------------------------------------------------------------------------------------------
        enc_inputs = []
        if "rgbOri" in self.enc_infos:
            enc_inputs.append(rgbOris) # rgbOri: [V,3,H,W] 这里的rgbOri是原图    
        
        if "camRay" in self.enc_infos:
            depths = F.interpolate(depths, size=self.inputSize, mode='bilinear', align_corners=False) # resize到dino原图大小，然后，下采样7倍到dino feature patch 大小
            normed_c2ws = self.normalize_c2ws(w2cs, T_norm, scale)  # 归一化 c2w
            ray_embed = self.get_ray_embedding_batch(intrinsic, normed_c2ws, h=self.inputSize[0], w=self.inputSize[1], depths=depths.squeeze(), opengl=True)
            enc_inputs.append(ray_embed)
        if "depth" in self.enc_infos:
            depths = F.interpolate(depths, size=self.inputSize, mode='bilinear', align_corners=False) # resize到dino原图大小，然后，下采样7倍到dino feature patch 大小
            enc_inputs.append(depths)
        if "normal" in self.enc_infos:
            enc_inputs.append(normals)
            # print("enc_inputs: ", enc_inputs[0].device, enc_inputs[0].shape, enc_inputs[1].device, enc_inputs[1].shape, enc_inputs[2].device, enc_inputs[2].shape)
                    
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            enc_inputs = torch.cat(enc_inputs, dim=1)    
            # print(f"enc_inputs: {enc_inputs.shape}, enc_inputs.dtype: {enc_inputs.dtype}")                        
            enc_feat = self.featEnc(enc_inputs) # [V,1,H,W] --> [V,1024,h,w]

            # doing back-projection
            intrinsic[:,0,0] /= scale # *w
            intrinsic[:,1,1] /= scale # *h
            intrinsic[:,0,:] *= self.featSize[1] # *w
            intrinsic[:,1,:] *= self.featSize[0] # *h
            
            depths = F.interpolate(depths, size=self.featSize, mode='bilinear', align_corners=False) # resize到dino feat 大小
            
            aggregated_feat, _ = self.batch_project_and_aggregate_chunkwise(attr_GS[..., :3], enc_feat, depths, intrinsic, w2cs, 
                                                                            T_norm=T_norm, depth_tol=self.acc)        
            torch.cuda.empty_cache()    
            
            feat_voxel = aggregated_feat
        # ------------------------------------------------------------------------------------------
        
        
        ''' 使用之前GS_decoder计算得到 voxel feature  '''
        # ------------------------------------------------------------------------------------------
        # # voxel_points  [N，3]
        # pos_voxel = voxel_points.unsqueeze(1).repeat(1,4,1) # 把voxel positions先expand重复4次
        # pos_voxel = pos_voxel.view(voxel_points.shape[0]*4, voxel_points.shape[-1])
        # num_voxel, dim_voxel = feat_voxel.shape
        # feat_voxel = feat_voxel.unsqueeze(1).repeat(1, 4, 1)  # 先加一个维度，然后沿着这个维度重复4次
        # feat_voxel = feat_voxel.view(-1, dim_voxel)
        
        # knn_idx = self.optimized_cdist_batch(pos_voxel, attr_GS[..., :3], k=16)
        # agg_feat_voxel = self.knn_average(feat_voxel, knn_idx)
        
        # x = torch.cat([feat_voxel, agg_feat_voxel, feat_GS], dim=-1)
        # ------------------------------------------------------------------------------------------
        
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
        delta_xyz = self.out_layer_xyz(h) * self.rep_config['lr']['_xyz']  
        delta_xyz = voxel_align_ratio * 0.5 * self.voxel_size * torch.tanh(delta_xyz)
        feats_xyz = delta_xyz + q_anchors
        feats_fdc = self.out_layer_fdc(h) * self.rep_config['lr']['_features_dc']
        feats_sca = self.out_layer_sca(h) * self.rep_config['lr']['_scaling']  
        feats_rot = self.out_layer_rot(h) * self.rep_config['lr']['_rotation'] 
        feats_opa = self.out_layer_opa(h) * self.rep_config['lr']['_opacity']  
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

