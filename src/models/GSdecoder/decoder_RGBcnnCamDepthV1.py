import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerBase, SparseTransformerCrossBase, TransformerBase, TransformerCrossBase
from ...representations import Gaussian
from einops import rearrange, repeat
# from torchvision.models import resnet50
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import open3d as o3d
# from torch_cluster import knn_graph
DEBUG=False
if DEBUG:
    out_dir = "outputs_TRELLISdev/test_results/test_differentiable_projection"
    os.makedirs(out_dir, exist_ok=True)

def print_memory_usage(tag=""):
    allocated = torch_npu.npu.memory_allocated() / 1024**2  # MB
    reserved = torch_npu.npu.memory_reserved() / 1024**2  # MB
    print(f"[{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def vis_uv_in_depth(depth, uv, savepath='debug_projection.png'):
    if depth is not None:
        depth_map_np = depth.cpu().numpy()
        if depth_map_np.ndim == 3:
            depth_map_np = depth_map_np.squeeze()
        uv_np = uv.cpu().numpy().squeeze() if uv is not None else None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(depth_map_np, cmap='gray')
        if uv_np is not None and len(uv_np) > 0:
            ax.scatter(uv_np[:, 0], uv_np[:, 1], s=5, c='red', marker='o', alpha=0.6)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)  # ✅ 释放资源

def vis_uv(points, savepath='debug_uv_points_plot.png'):
    # 转为 numpy 以便 matplotlib 使用
    points_np = points.cpu().numpy()

    # 分离 x 和 y 坐标
    x = points_np[..., 0]
    y = points_np[..., 1]

    # 画图
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, c='blue', alpha=0.6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # 图像坐标系 y 轴向下，需要反转一下y轴    
    plt.grid(False)
    plt.axis("equal")  # 保持比例一致
    plt.savefig(savepath, dpi=300, bbox_inches='tight')  # 可以改为 .pdf / .svg / .jpg 等
    plt.close()  # 释放内存        

from sklearn.decomposition import PCA
import numpy as np
import os
def visualize_feature_pca_all_shared(tensor: torch.Tensor, savepath:str='debug_features.png'):
    """
    可视化 V 个 feature map 的 PCA 降维结果，拼接成一张图保存。
    tensor: [V, C, H, W]
    save_dir: 保存目录
    prefix: 保存文件名前缀
    """
    V, C, H, W = tensor.shape
    device = tensor.device

    # [V, C, H, W] -> [V, H, W, C]
    feat = tensor.permute(0, 2, 3, 1).reshape(-1, C)  # [(V*H*W), C]

    # PCA 降维到3维 (RGB)
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat.detach().cpu().numpy())  # [(V*H*W), 3]
    feat_pca = torch.tensor(feat_pca, dtype=torch.float32).to(device)

    # normalize 到 [0, 1]
    feat_pca -= feat_pca.min(dim=0, keepdim=True)[0]
    feat_pca /= feat_pca.max(dim=0, keepdim=True)[0] + 1e-5

    # reshape 回图像: [V, H, W, 3] -> [V, 3, H, W]
    feat_img = feat_pca.view(V, H, W, 3).permute(0, 3, 1, 2)  # [V, 3, H, W]

    # 保存所有拼接图像
    save_image(feat_img, savepath)  # 每行放8张图
    print(f"[visualize_feature_pca_all_shared] Saved to {savepath}")

def vis_uv_in_depth(depth, uv_tensor, savepath='debug_projection_uv.png'):
    """
    Args:
        depth: [H, W] tensor or numpy array
        uv_tensor: [N, 2] or [V,N,2] normalized uv (in [-1,1])
    """
    if isinstance(depth, torch.Tensor):
        depth_np = depth.cpu().numpy()
    else:
        depth_np = depth

    H, W = depth_np.shape
    plt.figure(figsize=(8,6))
    plt.imshow(depth_np, cmap='gray')
    
    # 支持多个视角或单个 [N,2]
    if uv_tensor.dim() == 2:
        u_norm = uv_tensor[:, 0].cpu()
        v_norm = uv_tensor[:, 1].cpu()
        u = ((u_norm + 1) / 2) * (W - 1)
        v = ((v_norm + 1) / 2) * (H - 1)
        plt.scatter(u, v, s=5, c='red')
    elif uv_tensor.dim() == 3:
        for i in range(uv_tensor.shape[0]):
            u_norm = uv_tensor[i,:,0].cpu()
            v_norm = uv_tensor[i,:,1].cpu()
            u = ((u_norm + 1) / 2) * (W - 1)
            v = ((v_norm + 1) / 2) * (H - 1)
            plt.scatter(u, v, s=2, label=f'View {i}')
        plt.legend()

    plt.title('Projected UV on Depth')
    plt.axis('off')
    plt.savefig(savepath)
    plt.close()

def vis_visibility_mask(points, visibility_mask, savepath='visible_points.ply'):
    """
    Args:
        points: [N,3] 3D point cloud
        visibility_mask: [N] boolean
    Saves:
        .ply file showing visible (green) vs invisible (red) points
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    points_np = points.cpu().numpy()
    visible = visibility_mask.cpu().numpy()

    pcd.points = o3d.utility.Vector3dVector(points_np)
    colors = np.zeros_like(points_np)
    colors[visible] = [0,1,0]  # green = visible
    colors[~visible] = [1,0,0]  # red = invisible
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(savepath, pcd)

def vis_projected_color_on_depth(depth, uv_tensor, features_1024, savepath='projected_feature_on_depth.png'):
    """
    可视化高维特征（1024D），投影到深度图上。特征用 PCA 降维为 RGB。
    Args:
        depth: [H, W] tensor or numpy array
        uv_tensor: [N, 2] in normalized [-1,1]
        features_1024: [N, 1024] tensor
    """
    if isinstance(depth, torch.Tensor):
        depth_np = depth.cpu().numpy()
    else:
        depth_np = depth

    H, W = depth_np.shape
    N = features_1024.shape[0]

    # 1. 降维 [N,1024] -> [N,3]
    pca = PCA(n_components=3)
    features_np = features_1024.detach().cpu().numpy()
    colors_3d = pca.fit_transform(features_np)  # [N,3]
    colors_3d -= colors_3d.min()
    colors_3d /= (colors_3d.max() + 1e-6)  # normalize to [0,1]

    # 2. 像素坐标还原
    u_norm = uv_tensor[:, 0].cpu()
    v_norm = uv_tensor[:, 1].cpu()
    u = ((u_norm + 1) / 2) * (W - 1)
    v = ((v_norm + 1) / 2) * (H - 1)

    # 3. 可视化
    plt.figure(figsize=(8,6))
    plt.imshow(depth_np, cmap='gray')
    plt.scatter(u, v, c=colors_3d, s=10)
    plt.title('Projected High-Dim Feature (PCA RGB) on Depth')
    plt.axis('off')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
def vis_sampled_depth_on_image(depth, uv, sampled_depth, savepath='debug.png'):
    """
    depth: [H,W]
    uv: [N,2] in pixel coordinate (x,y)
    sampled_depth: [N]
    """
    import matplotlib.pyplot as plt
    import numpy as np

    depth_np = depth.cpu().numpy()
    uv_np = uv.cpu().numpy()
    d_np = sampled_depth.cpu().numpy()

    plt.figure(figsize=(10,6))
    plt.imshow(depth_np, cmap='gray')
    sc = plt.scatter(uv_np[:,0], uv_np[:,1], c=d_np, cmap='jet', s=3)
    plt.colorbar(sc)
    plt.savefig(savepath)
    plt.close()

def feature_to_color(feat):
    """将高维特征 [N, C] 映射为 RGB [N, 3]"""
    feat = feat.detach().cpu()
    feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-6)
    feat = feat[:, :3] if feat.shape[1] >= 3 else torch.nn.functional.pad(feat, (0, 3 - feat.shape[1]))
    feat = feat - feat.min()
    feat = feat / (feat.max() + 1e-6)
    return feat.numpy()

def to_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# def save_tensor_as_rgb(tensor, save_path, clamp_range=(-1, 1)):
#     """
#     tensor: [H, W, 3], torch.Tensor
#     """
#     tensor = tensor.detach().cpu().clamp(*clamp_range)
#     tensor = (tensor - clamp_range[0]) / (clamp_range[1] - clamp_range[0])  # Normalize to [0,1]
#     tensor = (tensor.numpy() * 255).astype(np.uint8)
#     plt.imsave(save_path, tensor)

def save_tensor_as_rgb(tensor, save_path):
    # tensor: [B, H, W, 3] or [ H, W, 3]
    if tensor.dim() == 3 and tensor.shape[-1] == 3:
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)  # [ H, W, 3] -> [1, 3, H, W]
    elif tensor.dim() == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)  # [B, 3, H, W]

    # 如果值在 [-1, 1]，则转为 [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)
    save_image(tensor, save_path)

def save_tensor_as_gray(tensor, save_path):
    """
    tensor: tensor: [B, 1, H, W] 或 [H, W] 或 [B, H, W], torch.Tensor
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1) if tensor.shape[0] != 1 else tensor.unsqueeze(0)
    elif tensor.dim() == 4 and tensor.shape[1] != 1:
        raise ValueError("Gray tensor must have shape [B, 1, H, W], [B, H, W], or [H, W]")

    tensor = tensor.clamp(0, 1)
    save_image(tensor, save_path)

def visualize_ray_field_grid(ray_embedding, stride=16, save_path="ray_field_grid.png", rows=4, cols=8):
    """
    将多个 ray embedding 可视化在一张图上（支持 B=32），每个子图是一个 quiver 图。
    
    Args:
        ray_embedding: [B, 6, H, W]
        stride: quiver 的稀疏程度
        save_path: 拼图保存路径
        rows, cols: 子图的行列数（默认 4x8=32）
    """
    B, C, H, W = ray_embedding.shape
    assert C == 6, "ray_embedding must have 6 channels: [rays_d(3), cross(3)]"
    assert B <= rows * cols, f"Batch size {B} exceeds grid capacity {rows * cols}"

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axs = axs.flatten()

    for b in range(B):
        rays_d = ray_embedding[b, :3]  # [3, H, W]
        ys, xs = torch.meshgrid(
            torch.arange(0, H, stride),
            torch.arange(0, W, stride),
            indexing='ij'
        )
        dx = rays_d[0, ys, xs]
        dy = -rays_d[1, ys, xs]  # 图像坐标下是正方向

        ax = axs[b]
        ax.quiver(xs.cpu(), ys.cpu(), dx.cpu(), dy.cpu(), scale=50, color='red')
        ax.invert_yaxis()
        ax.axis("off")

    # 清除多余子图
    for i in range(B, rows * cols):
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# def gumbel_softmax_sample(logits, temperature=1.0):
#     """
#     Sample from Gumbel-Softmax distribution with temperature
#     较低的 temperature 值会使得 选择变得更为“硬”，接近于 top-k 选择；较高的 temperature 会导致选择更加平滑
    
#     Args:
#         logits: [N_q, N_k], logits (attention weights)
#         temperature: Gumbel-Softmax temperature
#     Returns:
#         Sampled probabilities (soft mask): [N_q, N_k]
#     """
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#     y = logits + gumbel_noise
#     return torch.softmax(y / temperature, dim=-1)

def optimized_knn_batch(A: torch.Tensor, B: torch.Tensor, k: int, batch_size: int = 1024):
    N, _ = A.shape
    knn_idx = A.new_empty(N, k, dtype=torch.long)
    knn_dists = A.new_empty(N, k)

    A_norm = A.square().sum(1)
    B_norm = B.square().sum(1)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dist_matrix = A_norm[start:end, None] + B_norm - 2 * A[start:end] @ B.T
        knn_dists[start:end], knn_idx[start:end] = dist_matrix.topk(k, largest=False)

    return knn_dists.sqrt(), knn_idx

def optimized_cdist_batch(A: torch.Tensor, B: torch.Tensor, k: int, batch_size: int = 1536):
    N, _ = A.shape
    knn_idx = A.new_empty(N, k, dtype=torch.long)
    knn_dists = A.new_empty(N, k)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dist_matrix = torch.cdist(A[start:end], B)
        knn_dists[start:end], knn_idx[start:end] = dist_matrix.topk(k, largest=False)

    return knn_dists, knn_idx

# %%
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
        
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, use_checkpoint=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()
        self.use_checkpoint = use_checkpoint

    def forward_fn(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + self.drop_path(x)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self.forward_fn, x)
        else:
            return self.forward_fn(x)

class LightFeatureUNetV2(nn.Module):
    def __init__(self, in_chns=1, out_chns=1024, size=(292, 438)):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_chns, 64, kernel_size=3, stride=2, padding=1),   # 584x876 -> 292x438
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),       # 292x438 -> 146x219
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),      # 146x219 -> 146x219 (保持分辨率)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # self.down4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),      # 146x219 -> 146x219 (保持分辨率)
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),     # 146x219 -> 146x219 (保持分辨率)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Upsampling path with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1),  # 146x219 -> 146x219
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # self.up2 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=(0, 1)),  # 146x219 -> 292x438
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 146x219 -> 292x438
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(size=size, mode='bilinear', align_corners=False),  # 292x438 -> 292x438
            nn.Conv2d(256 + 64, out_chns, kernel_size=1)  # 融合 skip connection
        )

    def forward(self, x):
        # Down path
        x1 = self.down1(x)   # 292x438
        x2 = self.down2(x1)  # 146x219
        x3 = self.down3(x2)  # 146x219
        # x4 = self.down4(x3)  # 146x219

        # Bottleneck
        xb = self.bottleneck(x3)  # 146x219

        # Up path with skip connections
        xu1 = self.up1(xb)        # 146x219
        xu2 = self.up2(xu1)       # 292x438
        xu2 = torch.cat([xu2, x1], dim=1)  # skip connection

        out = self.up3(xu2)       # 292x438

        return out

class DecoderOnlyRGBCNNCamDepthInV1(nn.Module):
    """
    use normalTransformer
    replace backprojection with Cross attention
    use irregular sampling
    """
    def __init__(
        self,
        # voxel_size: float,  # 固定的体素大小
        model_channels: int,
        latent_channels: int,
        num_blocks_enc: int = 1,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope", "ptv3pe", "dual"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        num_hidden_dims: int = 4,
        init_ratio: float = 1.,
        voxel_mode: str = "center",
        dropout: float = 0.1, # << New: dropout rate
        resolution: int = 128,
        enc_infos=["camRay",], #  "camRay" "depth"
        img_resolution: tuple = (1168, 1752),
        if_useCA_forBP=False
        # voxel_bound: Optional[tuple] = (-0.025, 0.025),  # 默认为[-0.5, 0.5]的网格范围   
    ):
        super().__init__()
        
        self.if_useCA_forBP = if_useCA_forBP
        
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
        self.featSize =  (img_resolution[0]//2, img_resolution[1]//2) # (42, 63) #
        self.inputSize = img_resolution # (2352//4, 3528//4) # 原始输入DINO图像大小//4
        self.featEnc = LightFeatureUNetV2(in_chns=in_chns, out_chns=model_channels, size=self.featSize)
        self.featEnc.apply(init_weights)
        
        if self.if_useCA_forBP:
            # extract feature from xyz tp feature [N, 3]-->[N,1024]
            self.pointsEnc = nn.Linear(3, model_channels)            
            # cross-attention for 3D points filtering
            num_blocks_filterCA = 1 # TODO
            self.filterCA = TransformerCrossBase(
                in_channels=latent_channels,
                model_channels=model_channels,
                ctx_channels=model_channels,
                num_blocks=num_blocks_filterCA,
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
        
        # encoder for latent can be small
        # self.enc = SparseTransformerBase(
        #     in_channels=latent_channels,
        #     model_channels=model_channels,
        #     num_blocks=num_blocks_enc,
        #     num_heads=num_heads,
        #     num_head_channels=num_head_channels,
        #     mlp_ratio=mlp_ratio,
        #     attn_mode=attn_mode,
        #     window_size=window_size,
        #     pe_mode=pe_mode,
        #     use_fp16=use_fp16,
        #     use_bf16=use_bf16,
        #     use_checkpoint=use_checkpoint,
        #     qk_rms_norm=qk_rms_norm,
        #     dropout=dropout, # << New: dropout rate
        # )
        self.enc = TransformerBase(
            in_channels=model_channels,
            model_channels=model_channels,
            num_blocks=num_blocks_enc,
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
        
        self.rep_config = representation_config
        self.voxel_size = self.rep_config['voxel_size']   # 存储voxel_size
        
        # self.voxel_bound = voxel_bound
        if voxel_mode == "center":
            self.voxel_bound = (-self.voxel_size/2, self.voxel_size/2) 
        elif voxel_mode == "corner":
            self.voxel_bound = (0, self.voxel_size) 
        
        self.sh_degree = self.rep_config['sh_degree'] if 'sh_degree' in self.rep_config else 0
        self.num_features = 3 * ((self.sh_degree+1)**2)
        if self.sh_degree > 0 and '_features_rest' not in self.rep_config['lr']:
            self.rep_config['lr']['_features_rest'] = self.rep_config['lr']['_features_dc']
        
        # for PROJECTOR
        self.depth_min = 1e-6
        self.depth_max = 4.0
        self.acc = 4 / resolution
        
        self._calc_layout()
        
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
        
        self.out_layer_xyz = _build_outlayer(model_channels, self.rep_config['num_gaussians'] * 3, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_fdc = _build_outlayer(model_channels, self.rep_config['num_gaussians'] * self.num_features, 
                                             num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_sca = _build_outlayer(model_channels, self.rep_config['num_gaussians'] * 3, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_rot = _build_outlayer(model_channels, self.rep_config['num_gaussians'] * 4, num_hidden_dims=num_hidden_dims, mode='nn')
        self.out_layer_opa = _build_outlayer(model_channels, self.rep_config['num_gaussians'], num_hidden_dims=num_hidden_dims, mode='nn')
        
        # 'uniform', 'random', 'none'
        if self.rep_config['perturb_offset'] == 'uniform':
            self.offset_perturbation = init_ratio * self.uniform_sampling_cube()
        elif self.rep_config['perturb_offset'] == 'random':
            self._build_perturbation()
            self.offset_perturbation = init_ratio * self.offset_perturbation
            
        self.initialize_weights()
        all_modules = [self.enc] if not self.if_useCA_forBP else [self.enc, self.filterCA]
        for module in all_modules:
            if use_fp16:
                module.convert_to_fp16()
            if use_bf16:
                module.convert_to_bf16()

    def initialize_weights(self) -> None:
        all_modules = [self.enc] if not self.if_useCA_forBP else [self.enc, self.filterCA]
        for module in all_modules:
            module.initialize_weights()
            
        # # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        # nn.init.constant_(self.out_layer.bias, 0)
        
        # self-define  output layers
        all_mlps = [self.out_layer_xyz, self.out_layer_fdc, self.out_layer_sca, self.out_layer_rot, self.out_layer_opa]
        if self.if_useCA_forBP:
            all_mlps += [self.pointsEnc]
        for out_layer in all_mlps:
            # nn.init.constant_(out_layer.weight, 0)
            # nn.init.constant_(out_layer.bias, 0)
            
            for i, layer in enumerate(out_layer):
                if isinstance(layer, sp.SparseLinear):
                    if i == len(out_layer) - 1:
                        # 对最后一层使用 Zero 初始化
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    else:
                        # 对其他层使用 Kaiming 初始化
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])] # list:[ list:[] ], list1: len=self.rep_config['num_gaussians'], list2: len=3
        perturbation = torch.tensor(perturbation).float() * 2 - 1 # NOTE: 将pertubation normalize到 [-1,1] # shape: (self.rep_config['num_gaussians'], 3)
        self.offset_perturbation = torch.atanh(perturbation)


    def _calc_layout(self) -> None:
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_features_dc' : {'shape': (self.rep_config['num_gaussians'], 3), 
                              'size': self.rep_config['num_gaussians'] * 3}
        }
        if self.sh_degree: self.layout['_features_rest'] = {'shape': (self.rep_config['num_gaussians'], self.num_features - 3),
                                                            'size': self.rep_config['num_gaussians'] * (self.num_features - 3)}
        start = 0
        for v in self.layout.values():
            v['range'] = (start, start + v['size'])
            start += v['size']
        # self.out_channels = start

    def to_representation(self, x: torch.Tensor, 
                          voxel_points=None, 
                          voxel_align_ratio=1) -> Gaussian: # aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            representations
        """        
        # for i in range(x.shape[0]):
        representation = Gaussian(
            sh_degree=self.sh_degree,
            mininum_kernel_size = self.rep_config['3d_filter_kernel_size']*0.0001,
            scaling_bias = self.rep_config['scaling_bias'],
            opacity_bias = self.rep_config['opacity_bias'],
            scaling_activation = self.rep_config['scaling_activation']
        )
        # `x.coords` 是稀疏张量中非零元素的坐标
        for k, v in self.layout.items():
            if k == '_xyz':
                
                if voxel_points is not None:
                    xyz = voxel_points.to(x.device)
                else:
                    raise NotImplementedError("voxel_points should be provided as PE")

                offset = x[:, v['range'][0]:v['range'][1]].reshape(x.shape[0], *v['shape'])
                offset = offset * self.rep_config['lr'][k]   
                if self.rep_config['perturb_offset'] != 'none':
                    offset = offset + self.offset_perturbation.to(offset.device)
                offset = torch.tanh(offset) * 0.5 * self.voxel_size
                _xyz = xyz.unsqueeze(1) + voxel_align_ratio*offset

                setattr(representation, k, _xyz.flatten(0, 1))
                
            else:
                feats = x[:, v['range'][0]:v['range'][1]].reshape(x.shape[0], *v['shape']).flatten(0, 1)
                if k in ['_features_dc', '_features_rest']:
                    feats = feats.reshape(feats.shape[0], -1, 3)
                feats = feats * self.rep_config['lr'][k]
                setattr(representation, k, feats)

        # ret.append(representation)
        return representation
    
    @staticmethod
    def batch_project_and_aggregate(points, features, depths, intrinsics, w2cs, T_norm=None, depth_tol=0.05,
                                    fill_empty=True, knn_k=8, knn_eps=1e-6):
        """
        Args:
            points: [N, 3] world coordinates of 3D points
            features: [V, C, H, W] feature maps from V views
            depths: [V, H, W] depth maps from V views (depth unit must match camera coord depth)
            intrinsics: [V, 3, 3] camera intrinsics
            w2cs: [V, 4, 4] world-to-camera extrinsics
            depth_tol: float, max allowed depth difference for visibility
            fill_empty: bool, 是否用邻近点插值填充无特征点
            knn_k: int, 邻居数量
            knn_eps: float, 防止除零的小数
        
        Returns:
            aggregated_feats: [N, C] aggregated features per 3D point
            visibility_mask: [N] boolean mask indicating if point is visible in any view
        """
        device = points.device
        V, C, H, W = features.shape
        N = points.shape[0]

        # 1. 转为齐次坐标 [N,4]
        homog_points = torch.cat([points, torch.ones((N,1), device=device)], dim=-1)

        # 2. 世界坐标 -> 相机坐标 [V,N,4]
        cam_points = torch.einsum('vij,nj->vni', w2cs, homog_points)
        cam_points_xyz = cam_points[..., :3]  # [V,N,3]

        # 3. 判断点是否在相机前方
        z_cam = cam_points_xyz[..., 2]  # [V,N]
        # if T_norm is not None:
        #     z_cam *= T_norm[2,2]
        in_front = z_cam > 1e-6

        # 防止除零，做归一化投影坐标
        z_safe = z_cam.clamp(min=1e-6)
        x_norm = cam_points_xyz[..., 0] / z_safe
        y_norm = cam_points_xyz[..., 1] / z_safe

        # 4. 提取内参
        fx = intrinsics[:, 0, 0].view(V,1)
        fy = intrinsics[:, 1, 1].view(V,1)
        cx = intrinsics[:, 0, 2].view(V,1)
        cy = intrinsics[:, 1, 2].view(V,1)

        # 5. 投影到像素坐标
        u = fx * x_norm + cx  # [V,N]
        v = fy * y_norm + cy  # [V,N]

        # 6. 判断是否在图像范围内
        valid_u = (u >= 0) & (u <= (W - 1))
        valid_v = (v >= 0) & (v <= (H - 1))
        in_img = valid_u & valid_v  # [V,N]

        # 7. 归一化到[-1,1]，准备grid_sample
        u_norm = 2.0 * (u / (W - 1)) - 1.0  # [V,N]
        v_norm = 2.0 * (v / (H - 1)) - 1.0  # [V,N]
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2).to(depths.device)  # [V,N,1,2]

        # 8. 从深度图采样对应深度值
        sampled_depths = F.grid_sample(
            depths.contiguous(),  # [V,1,H,W]
            grid.contiguous(),  # [V,N,12]
            mode='bilinear',
            align_corners=True
        ).squeeze(1).squeeze(-1)  # [V,N]

        # 9. 深度有效性：采样深度有效且与点深度差异小于阈值
        depth_positive = sampled_depths > 0
        # print(f"[debug] sampled_depths: {sampled_depths.shape, sampled_depths[0].max(), sampled_depths[1].max()}, z_cam: {z_cam.shape, z_cam[0].max(), z_cam[1].max()}")
        depth_close = torch.abs(sampled_depths - z_cam) < depth_tol
        depth_valid = depth_positive & depth_close  # [V,N]

        # 10. 综合有效性掩码
        valid_mask = in_front & in_img & depth_valid  # [V,N]

        # 11. 从 feature maps 采样特征
        sampled_feats = F.grid_sample(
            features.contiguous(),  # [V,C,H,W]
            grid.contiguous(),  # [V,N,1,2]
            mode='bilinear',
            align_corners=True
        ).squeeze(-1).permute(2, 0, 1)  # [N,V,C]
        
        # if DEBUG:
        #     for vi in range(depths.shape[0]):
        #         vis_uv_in_depth(depths[vi].squeeze(), grid[vi].squeeze(),  os.path.join(out_dir, f'debug_view{vi}_uv_on_depth.png'))
        #         vis_visibility_mask(points, valid_mask[vi],  os.path.join(out_dir, f'debug_view{vi}_visible_points.ply'))
        #         vis_projected_color_on_depth(depths[vi].squeeze(), grid[vi].squeeze(), sampled_feats[:,vi,:],  os.path.join(out_dir, f'debug_view{vi}_colored_proj.png'))

        # 12. 加权平均
        weights = valid_mask.float().permute(1, 0)  # [N,V]
        total_weights = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weighted_feats = sampled_feats * weights.unsqueeze(-1)  # [N,V,C]
        aggregated_feats = weighted_feats.sum(dim=1) / total_weights  # [N,C]

        # 13. 是否至少在一个视角可见
        visibility_mask = weights.sum(dim=1) > 0  # [N]
        
        # if DEBUG:
        #     color_before = feature_to_color(aggregated_feats)
        #     pcd_before = to_pcd(points, color_before)
        #     o3d.io.write_point_cloud(os.path.join(out_dir, f'debug_features_before_fill.ply'), pcd_before)
        
        # 15. 对无特征点做KNN插值补全
        if fill_empty:
            
            missing_mask = ~visibility_mask  # 无特征点mask
            if missing_mask.any():

                # 源点（有特征）和目标点（无特征）
                src_points = points[visibility_mask]  # [M,3]
                src_feats = aggregated_feats[visibility_mask]  # [M,C]
                tgt_points = points[missing_mask]  # [K,3]

                if src_points.shape[0] >= knn_k:
                    # 计算距离（这里用 torch.cdist）
                    # dists = torch.cdist(tgt_points, src_points, p=2)  # [K,M]
                    # knn_dists, knn_idx = torch.topk(dists, knn_k, largest=False, dim=1)  # [K,knn_k]
                    # breakpoint()
                    knn_dists, knn_idx = optimized_cdist_batch(tgt_points, src_points, knn_k)

                    # 权重反比距离
                    knn_weights = 1.0 / (knn_dists + knn_eps)
                    knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # 归一化

                    # 加权插值
                    interpolated_feats = torch.sum(src_feats[knn_idx] * knn_weights.unsqueeze(-1), dim=1)  # [K,C]
                    
                    # 填回去
                    aggregated_feats[missing_mask] = interpolated_feats
                            
        # if DEBUG:
        #     color_after = feature_to_color(aggregated_feats)
        #     pcd_after = to_pcd(points, color_after)
        #     o3d.io.write_point_cloud(os.path.join(out_dir, f'debug_features_after_fill.ply'), pcd_after)

        return aggregated_feats, visibility_mask
    
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
                    depths_chunk.float(),  # [V_c,1,H,W]
                    grid.float(),                      # [V_c,N_c,1,2]
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1).squeeze(-1).bfloat16()  # [V_c, N_c]
                
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
                    features_chunk.float(),          # [V_c, C, H, W]
                    grid.float(),                   # [V_c, N_c, 1, 2]
                    mode='bilinear',
                    align_corners=True
                ).squeeze(-1).bfloat16()              # [V_c, C, N_c]
                sampled_feats = sampled_feats.permute(2, 0, 1)  # [N_c, V_c, C]
                
                
                # if DEBUG:
                #     for vi in range(V_c):
                #         n_start_str = str(n_start).zfill(4)
                #         v_start_str = str(v_start).zfill(4)
                #         vis_uv_in_depth(depths_chunk[vi].squeeze(), grid[vi].squeeze(), 
                #                         os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_uv_on_depth.png'))
                #         vis_visibility_mask(points_chunk, valid_mask[vi], 
                #                             os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_visible_points.ply'))
                #         vis_projected_color_on_depth(depths_chunk[vi].squeeze(), grid[vi].squeeze(), 
                #                                      sampled_feats[:, vi, :], 
                #                                       os.path.join(out_dir, f'debug_{n_start_str}_{v_start_str}_view{vi}_colored_proj.png'))
                
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
        
        # if DEBUG:
        #     color_before = feature_to_color(aggregated_feats)
        #     pcd_before = to_pcd(points, color_before)
        #     o3d.io.write_point_cloud(os.path.join(out_dir, f'debug_features_before_fill.ply'), pcd_before)

        # 对无特征点做KNN插值补全
        if fill_empty:
            missing_mask = ~visibility_mask  # 无特征点mask
            if missing_mask.any().item():
                src_points = points[visibility_mask]  # [M,3]
                src_feats = aggregated_feats[visibility_mask]  # [M,C]
                tgt_points = points[missing_mask]  # [K,3]

                k = min(knn_k, src_points.shape[0])
                # dists = torch.cdist(tgt_points, src_points, p=2)  # [K, M]
                # knn_dists, knn_idx = torch.topk(dists, k, largest=False, dim=1)
                knn_dists, knn_idx = optimized_cdist_batch(tgt_points, src_points, k)

                knn_weights = 1.0 / (knn_dists + knn_eps)
                knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # 归一化

                interpolated_feats = torch.sum(src_feats[knn_idx] * knn_weights.unsqueeze(-1), dim=1)  # [K,C]
    
                aggregated_feats[missing_mask] = interpolated_feats

        # if DEBUG:
        #     color_after = feature_to_color(aggregated_feats)
        #     pcd_after = to_pcd(points, color_after)
        #     o3d.io.write_point_cloud( os.path.join(out_dir, f'debug_features_after_fill.ply'), pcd_after)
        
        torch_npu.npu.empty_cache()

        return aggregated_feats, visibility_mask
    
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
        
        if DEBUG:
            # Save for each image in batch
            print("rays_d center: ", rays_d[:, h//2, 2//2])  # 应该接近 [0, 0, 1]（OpenCV）或 [0, 0, -1]（OpenGL）
            lengths = torch.norm(rays_d[:, h//2, 2//2], dim=-1)
            print(lengths)  # 理想情况下接近1
            center_cross = torch.cross(rays_o[:, h//2, w//2], rays_d[:, h//2, w//2])
            print("Cross at center:", center_cross)
            save_tensor_as_rgb(rays_d, os.path.join(out_dir, f"rays_direction.png"))
            save_tensor_as_rgb(cross, os.path.join(out_dir, f"rays_cross.png"))
            save_tensor_as_gray(mask.float(), os.path.join(out_dir, f"mask.png"))
            visualize_ray_field_grid(ray_embedding, save_path=os.path.join(out_dir, f"ray_field_grid.png"), rows=4, cols=8)  # 使用默认 stride=16

        return ray_embedding
    
    @staticmethod
    def gen_uniform_n_points(n=10, local_size=0.5):
        """
        支持任意 n，不要求是立方数
        """
        if n == 1:
            return torch.zeros((n,3))
        k = int(torch.ceil(torch.tensor(n ** (1/3))))  # 向上取整，例如 n=10 → k=3 → 27个点
        lin = torch.linspace(-0.5, 0.5, k)
        grid = torch.stack(torch.meshgrid([lin, lin, lin], indexing='ij'), dim=-1)  # [k, k, k, 3]
        grid = grid.reshape(-1, 3) * local_size  # [k³, 3]
        grid = grid[:n]  # 截取前 n 个
        return grid

    # @staticmethod
    # def normalize_c2ws(ori_w2cs: torch.Tensor, T_norm: torch.Tensor) -> torch.Tensor:
    #     """
    #     将原始 w2c 外参变换到归一化空间，输出归一化后的 c2w，并修正旋转部分为正交矩阵。

    #     Args:
    #         ori_w2cs: Tensor of shape [B, 4, 4], 原始世界到相机的变换矩阵
    #         T_norm:   Tensor of shape [4, 4], 世界坐标归一化矩阵（用于归一化 mesh）

    #     Returns:
    #         normed_c2ws: Tensor of shape [B, 4, 4], 归一化坐标下的相机位姿 c2w（旋转已单位化）
    #     """

    #     assert ori_w2cs.ndim == 3 and ori_w2cs.shape[1:] == (4, 4), "Expected ori_w2cs to be [B, 4, 4]"
    #     assert T_norm.shape == (4, 4), "Expected T_norm to be [4, 4]"

    #     B = ori_w2cs.shape[0]

    #     # 计算归一化场景下的 c2w： T_norm @ inverse(w2c)
    #     c2ws_orig = torch.linalg.inv(ori_w2cs)  # [B, 4, 4]
    #     T_norm = T_norm.to(ori_w2cs.device)
    #     normed_c2ws = T_norm @ c2ws_orig  # [B, 4, 4]

    #     # 单位正交化旋转部分
    #     R = normed_c2ws[:, :3, :3]  # [B, 3, 3]
    #     U, _, Vt = torch.linalg.svd(R)  # 分别是 [B, 3, 3]

    #     # 修正 det(U @ Vt) 为正：避免反射
    #     # 计算 det(U @ Vt) = det(U) * det(Vt)
    #     det_U = torch.det(U)  # [B]
    #     det_Vt = torch.det(Vt)  # [B]
    #     det = (det_U * det_Vt).sign().view(B, 1, 1)  # [B,1,1]

    #     # 修正 Vt：乘以 diag(1,1,det)
    #     Vt_fixed = Vt.clone()
    #     Vt_fixed[:, 2, :] = Vt_fixed[:, 2, :] * det.view(B, 1)

    #     R_fixed = U @ Vt_fixed  # [B, 3, 3]

    #     # 替换原始矩阵中的旋转部分
    #     normed_c2ws[:, :3, :3] = R_fixed

    #     # cam_positions = normed_c2ws[:, :3, 3]
    #     # print("Camera positions (in normalized world coordinates):", cam_positions)

    #     return normed_c2ws

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
    def sparse_attention_with_gumbel(attn_weights, v, topk=None, temp=0.1, thresh=0.01, 
                                     std_threshold=1e-3, min_attention=1e-5):
        """
        Perform sparse attention aggregation using Gumbel-Softmax sampling.
        Args:
            attn_weights: [N_q, N_k], attention weights between q and k/v
            v: [N_k, d], value features (k and v are the same in your case)
            topk: number of top-k attention weights to keep (optional)
            temp: temperature for Gumbel-Softmax
            thresh: threshold to consider for sparsity (for mask)
            std_threshold: threshold for standard deviation of attention weights to filter out non-informative queries
            min_attention: threshold to discard queries with very low total attention
        Returns:
            weighted_v: [M, d], aggregated features from relevant kv
            valid_q_mask: [N_q] bool, mask indicating which queries are kept
        """
        N_q, N_k = attn_weights.shape
        d = v.shape[1]

        # 1. Gumbel-Softmax sampling to generate soft attention mask
        soft_mask = gumbel_softmax_sample(attn_weights, temperature=temp)  # [N_q, N_k]

        # 2. Optional: Top-K pruning (if required)
        if topk is not None:
            topk_vals, topk_idx = torch.topk(soft_mask, topk, dim=1)
            sparse_mask = torch.zeros_like(soft_mask)
            sparse_mask.scatter_(1, topk_idx, 1.0)
            soft_mask = sparse_mask  # Using hard threshold based on top-k
        else:
            # Optional: Apply a threshold for sparsity
            soft_mask = (soft_mask > thresh).float()

        # 3. Compute q_importance (total attention for each query)
        q_importance = soft_mask.sum(dim=1)  # Total attention for each query

        # 4. Compute the standard deviation of the attention weights for each query
        q_importance_std = soft_mask.std(dim=1)  # Standard deviation of attention for each query

        # 5. Filter out queries based on standard deviation (those with very low std are unimportant)
        valid_q_mask_std = q_importance_std > std_threshold

        # 6. Filter out queries with very low attention (based on q_importance)
        valid_q_mask_attention = q_importance > min_attention

        # 7. Combine the masks: Keep only queries that have both high std and high total attention
        valid_q_mask = valid_q_mask_std & valid_q_mask_attention

        # 8. Aggregate values based on soft mask and attention weights (instead of just soft mask)
        weighted_v_all = (soft_mask * attn_weights) @ v  # [N_q, d]

        # 9. Get the aggregated features for valid queries
        weighted_v = weighted_v_all[valid_q_mask]  # [M, d]

        return weighted_v, valid_q_mask


    def forward(self, feats=None, rgbOris=None, depths=None, intrinsic=None,
                ori_w2cs=None, w2cs=None, scale=None, T_norm=None, normals=None,
                voxel_points=None, voxel_align_ratio=5., grid_origin=None, if_fix_xyz=False) -> List[Gaussian]:
        """
        depths : [V, H, W]
        voxel_points: [N, 3]
        intrinsic: [V,3,3] 这里的intrinsics是normalized之后的内参, 也就是与图像大小无关的内参矩阵
        """        
        torch_npu.npu.empty_cache()
        # torch_npu.npu.ipc_collect()  # 收集跨进程的缓存
        enc_inputs = []
        
        # intrinsic[:,0,0] /= scale # *w
        # intrinsic[:,1,1] /= scale # *h
        
        # if "rgbFeats" in self.enc_infos:
        #     enc_inputs.append(rgb_feats)
        
        if "rgbOri" in self.enc_infos:
            enc_inputs.append(rgbOris) # rgbOri: [V,3,H,W] 这里的rgbOri是原图    
        
        if "camRay" in self.enc_infos:
            depths = F.interpolate(depths, size=self.inputSize, mode='bilinear', align_corners=False) # resize到dino原图大小，然后，下采样7倍到dino feature patch 大小
            normed_c2ws = self.normalize_c2ws(w2cs, T_norm, scale)  # 归一化 c2w
            
            # normed_c2ws = T_norm.to(ori_w2cs.device) @ torch.linalg.inv(ori_w2cs) 
            # # normed_c2ws = torch.inverse(T_norm).to(ori_w2cs.device) @ torch.inverse(ori_w2cs) # torch.inverse(w2cs)
            ray_embed = self.get_ray_embedding_batch(intrinsic, normed_c2ws, h=self.inputSize[0], w=self.inputSize[1], depths=depths.squeeze(), opengl=True)
            enc_inputs.append(ray_embed)
        if "depth" in self.enc_infos:
            depths = F.interpolate(depths, size=self.inputSize, mode='bilinear', align_corners=False) # resize到dino原图大小，然后，下采样7倍到dino feature patch 大小
            enc_inputs.append(depths)
        if "normal" in self.enc_infos:
            enc_inputs.append(normals)
            # print("enc_inputs: ", enc_inputs[0].device, enc_inputs[0].shape, enc_inputs[1].device, enc_inputs[1].shape, enc_inputs[2].device, enc_inputs[2].shape)
                
        if len(enc_inputs) > 0:
            
            with torch.autocast(device_type='npu', dtype=torch.bfloat16):

                enc_inputs = torch.cat(enc_inputs, dim=1)    
                # print(f"enc_inputs: {enc_inputs.shape}, enc_inputs.dtype: {enc_inputs.dtype}")                        
                enc_feat = self.featEnc(enc_inputs) # [V,1,H,W] --> [V,1024,h,w]
                # if DEBUG:
                #     save_image(depths, "debug_depth.png")       
                #     visualize_feature_pca_all_shared(enc_feat, "debug_feature.png" ) 

                if not self.if_useCA_forBP: 
                    # doing back-projection
                    intrinsic[:,0,0] /= scale # *w
                    intrinsic[:,1,1] /= scale # *h
                    intrinsic[:,0,:] *= self.featSize[1] # *w
                    intrinsic[:,1,:] *= self.featSize[0] # *h

                    depths = F.interpolate(depths, size=self.featSize, mode='bilinear', align_corners=False) # resize到dino feat 大小

                    aggregated_feat, _ = self.batch_project_and_aggregate_chunkwise(voxel_points, enc_feat, depths, intrinsic, w2cs, T_norm=T_norm, depth_tol=self.acc)        
                    # aggregated_feat, _ = self.batch_project_and_aggregate(voxel_points, enc_feat, depths, intrinsic, w2cs, T_norm=T_norm, depth_tol=self.acc)
                    torch_npu.npu.empty_cache()    
                    
                    x = aggregated_feat if feats is None else feats + aggregated_feat

                else:
                    # use cross attention to replace back-projection
                    x = enc_feat.squeeze().reshape(1024, -1).t()
                    # extract position feature 
                    voxel_points_feats = self.pointsEnc(voxel_points) # [N, 3] --> [N, 1024]
                    # doing Cross-Attention 
                    x = self.filterCA.forward(voxel_points_feats.unsqueeze(0), x.unsqueeze(0), coords=voxel_points, coords_cxt=None, return_scores=False)
        
            
        else:
            x = feats
        
        # x = x.to(next(self.enc.parameters()).dtype)
        # breakpoint()

        # doing Self-Attention for latent tensor
        h = self.enc.forward(x.unsqueeze(0), voxel_points) # h.feats: [11887, 8], h.coords:  [11887, 4]
        # print("h: ", h.shape)
        # h = h.to(x.dtype) # h = h.type(x.dtype) # h.feats: [11887, 768]
        # h.feats = h.feats.to(x.dtype)
        h = F.layer_norm(h, h.shape[-1:]) # h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        # MLP from 
        # h = self.out_layer(h) # h.feats: [11887, 7168]
        feats_xyz = self.out_layer_xyz(h).float() # h.feats: [11887, K*3]
        feats_fdc = self.out_layer_fdc(h).float() # h.feats: [11887, K*3]
        feats_sca = self.out_layer_sca(h).float() # h.feats: [11887, K*3]
        feats_rot = self.out_layer_rot(h).float() # h.feats: [11887, K*4]
        feats_opa = self.out_layer_opa(h).float() # h.feats: [11887, K*1]
        new_feats = torch.cat([feats_xyz, feats_opa, feats_sca, feats_rot, feats_fdc], dim=-1)  # [11887, K*14]
        # new_feats = torch.cat([feats_xyz, feats_fdc, feats_sca, feats_rot, feats_opa], dim=-1)  # [11887, K*14]
        # h = new_feats # h = h.replace(feats=new_feats, coords=h.coords)
        return [self.to_representation(new_feats, 
                                       voxel_points=voxel_points,
                                       voxel_align_ratio=voxel_align_ratio)] # h.feats: [11887, 7168], h.coords: [11887, 4] 
