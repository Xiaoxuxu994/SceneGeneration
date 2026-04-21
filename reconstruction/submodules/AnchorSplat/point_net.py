from functools import partial
import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential
from torch.utils.checkpoint import checkpoint_sequential

try:
    from torch_npu import npu_fusion_attention
    NPU_AVAILABLE = True
except:
    from torch.nn.functional import scaled_dot_product_attention
    NPU_AVAILABLE = False

from submodules.AnchorSplat.utils.attention import Attention
from submodules.AnchorSplat.utils.block import Block
from submodules.AnchorSplat.point_sampler import contract, RADIUS


class SimpleAttention(Attention):
    def __init__(self, dim: int, num_heads: int, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        if NPU_AVAILABLE: self.spda = lambda q, k, v: npu_fusion_attention(
            q, k, v,
            head_num=num_heads,
            input_layout='BSH',
            scale=self.scale
        )[0]
        else: self.spda = lambda q, k, v: scaled_dot_product_attention(
            q.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2),
            k.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2),
            v.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        ).transpose(1, 2).flatten(2)

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        q, k, v = self.qkv.forward(x).chunk(3, 2)
        x = self.spda(q, k, v)
        x = self.proj(x)
        return x


class SampleAttention(SimpleAttention):
    def __init__(self, dim: int, num_heads: int, num_kv: int, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.num_kv = num_kv

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        q, k, v = self.qkv.forward(x).chunk(3, 2)
        x = self.spda(q, k[:, :self.num_kv], v[:, :self.num_kv])
        x = self.proj(x)
        return x


class PointBackbone(Module):
    def __init__(self, dim_in: int, dim: int, num_heads: int, depth: int, num_kv: int,
                 num_pe_freqs: int, pe_contract: bool = False, checkpoint: bool = True):
        super().__init__()
        self.feat_proj = nn.Linear(dim_in, dim) if dim_in != dim else nn.Identity()
        freqs = torch.logspace(0, num_pe_freqs - 1, num_pe_freqs, 2)
        self.freqs = nn.Buffer(torch.pi / 2 / RADIUS * freqs)
        self.coord_proj = nn.Linear(3 + 6 * num_pe_freqs, dim)
        self.pe_contract = pe_contract  # whether to apply PE on contracted coordinates
        self.depth = depth
        attn = partial(SampleAttention, num_kv=num_kv)
        self.blocks = Sequential(*[Block(dim, num_heads, attn_class=attn) for _ in range(depth)])
        self.checkpoint = checkpoint

    def forward(self, feature: Tensor, coords: Tensor, voxel_sizes: Tensor = None) -> Tensor:
        if self.pe_contract: coords = contract(coords)  # norm within RADIUS
        embed = (coords[..., None] * self.freqs).flatten(2)
        embed = torch.cat((coords, embed.cos(), embed.sin()), 2)
        feature = self.feat_proj.forward(feature) + self.coord_proj.forward(embed)
        return checkpoint_sequential(self.blocks, self.depth, feature, False) \
            if self.checkpoint else self.blocks(feature)


# class Refiner(nn.Module):
#     sampler: VoxelSampler
#     head: BaseGaussianHead

#     def __init__(self, backbone: dict, mlp: dict):
#         super().__init__()
#         feat_cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
#         layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
#         self.feat_cnn = nn.Sequential(*(getattr(feat_cnn, l) for l in layers))
#         self.num_kv = backbone['num_kv']
#         self.backbone: PointBackbone = create_object(OmegaConf.create(backbone))
#         self.mlp: GaussianMLP = create_object(OmegaConf.create(mlp))

#     def forward(self, gs: dict[str, Tensor], gs_kv: dict[str, Tensor], render: Tensor,
#                 images: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor,
#                 voxel_sizes: Tensor) -> dict[str, Tensor]:
#         # sample kv coordinates
#         coords = contract(gs.mean) if self.sampler.contract_scene else gs.mean
#         anchors, _ = self.sampler.sample_batch(coords, self.num_kv)
#         coords = torch.cat((anchors, coords), 1)

#         # pointwise render error feature
#         error = (images - render).movedim(2, -1)
#         error = aggregate_feature(coords, error, depth, intrinsics, extrinsics)
#         feature = self.feat_cnn(images.flatten(0, 1)) - self.feat_cnn(render.flatten(0, 1))
#         feature = feature.movedim(1, -1).unflatten(0, images.shape[:2])
#         feature = aggregate_feature(coords, feature, depth, intrinsics, extrinsics)
#         gs_vals = torch.cat(list(gs.values()), 2).to(feature.dtype)
#         gs_kv_vals = torch.cat(list(gs_kv.values()), 2).to(feature.dtype)
#         feature = torch.cat((torch.cat((gs_kv_vals, gs_vals), 1), error, feature), 2)

#         # GS update
#         update = self.backbone.forward(feature, coords, voxel_sizes)
#         update = self.mlp.forward(torch.cat((gs_vals, update[:, self.num_kv:]), 2))
#         update = update.split_with_sizes([v.shape[2] for v in gs.values()], 2)
#         # return {k: self.head.act[k](x + self.head.inv_act[k](v, voxel_sizes), voxel_sizes)
#         #         for x, (k, v) in zip(update, gs.items())}
#         offset_max = self.head.offset_max * voxel_sizes
#         scale_max = self.head.scale_max * voxel_sizes
#         gs.mean = gs.mean + clamp(update[0], -offset_max, offset_max)
#         gs.opacity = clamp(gs.opacity + update[1], 0, 1)
#         gs.scale = clamp(gs.scale + update[2], torch.zeros_like(scale_max), scale_max)
#         gs.rotation = nn.functional.normalize(gs.rotation + update[3], dim=-1)
#         gs.color = gs.color + update[4] * self.head.sh_mask
#         return gs


# class PatchBackbone(nn.Module):
#     def __init__(self, dim_in: int, dim: int, num_heads: int, depth: int, num_kv: int,
#                  num_pe_freqs: int, pe_contract: bool = False, checkpoint: bool = True):
#         super().__init__()

#     def forward(self, feature: Tensor, coords: Tensor, voxel_sizes: Tensor) -> Tensor:
#         pass


# class PatchAttention(nn.Module):
#     def __init__(self,
#                 in_channels: int,
#                 model_channels: int,
#                 num_blocks: int,
#                 num_heads: Optional[int] = None,
#                 num_head_channels: Optional[int] = 256,
#                 mlp_ratio: float = 4.0,
#                 attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
#                 window_size: Optional[int] = None,
#                 pe_mode: Literal["ape", "rope", "ptv3pe", "ptv3rope", "dual"] = "ape",
#                 use_fp16: bool = False,
#                 use_bf16: bool = False,
#                 use_checkpoint: bool = False,
#                 qk_rms_norm: bool = False,
#                 dropout: float = 0.1,   # << New: dropout rate
#                 backend: str = 'flash_attn_npu',
#                 voxel_size: float = 0.001,
#                 patch_size: int = 64
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.num_blocks = num_blocks
#         self.window_size = window_size
#         self.num_heads = num_heads # or model_channels // num_head_channels
#         self.mlp_ratio = mlp_ratio
#         self.attn_mode = attn_mode
#         self.pe_mode = pe_mode
#         self.use_checkpoint = use_checkpoint
#         self.qk_rms_norm = qk_rms_norm
#         self.backend = backend
#         self.voxel_size = voxel_size
#         self.patch_size = patch_size
#         self.num_head_channels = num_head_channels

#         if pe_mode == "ape":
#             self.pos_embedder = AbsolutePositionEmbedder(model_channels)
#         elif self.pe_mode == "ptv3pe":
#             self.pos_embedder = PTV3PosEmbedding(hidden_dim=192, dim=model_channels)
#         elif self.pe_mode == "dual":
#             self.pos_embedder = DualPathPositionalEncoding(max_voxel_dim=128,
#                                                         index_embed_dim=128,      # per axis
#                                                         coord_embed_dim=model_channels-128*3, # 128*3 + 640 = 1024
#                                                         coord_type='fourier',     # or 'mlp'
#                                                         num_freqs=16              # optional
#                                                     )
#         elif self.pe_mode == "ptv3rope":
#             self.pos_embedder = PTV3RopeEmbedding(hidden_dim=192, dim=model_channels,
#                                                   act_layer=nn.LeakyReLU)
#         elif self.pe_mode == 'rope':
#             pass
#         else:
#             raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")

#         if use_bf16:
#             self.dtype = torch.bfloat16
#         elif use_fp16:
#             self.dtype = torch.float16
#         else:
#             self.dtype = torch.float32


#         self.input_layer = nn.Linear(in_channels, model_channels)

#         # Multi-head attention for patch-to-patch and within-patch self-attention
#         self._init_transformer_blocks(model_channels, pe_mode, backend)

#     def _init_transformer_blocks(self, model_channels, pe_mode, backend):
#         self.patch_blocks = nn.ModuleList([
#             TransformerBlock(
#                 model_channels,
#                 num_heads=self.num_heads,
#                 mlp_ratio=self.mlp_ratio,
#                 attn_mode=attn_mode,
#                 window_size=window_size,
#                 shift_window=shift_window,
#                 use_checkpoint=self.use_checkpoint,
#                 use_rope=(pe_mode == "rope"),
#                 qk_rms_norm=self.qk_rms_norm,
#                 backend=backend,
#             )
#             for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
#         ])

#         self.global_blocks = nn.ModuleList([
#             TransformerBlock(
#                 model_channels,
#                 num_heads=self.num_heads,
#                 mlp_ratio=self.mlp_ratio,
#                 attn_mode=attn_mode,
#                 window_size=window_size,
#                 shift_window=shift_window,
#                 use_checkpoint=self.use_checkpoint,
#                 use_rope=(pe_mode == "rope"),
#                 qk_rms_norm=self.qk_rms_norm,
#                 backend=backend,
#             )
#             for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
#         ])

#     @property
#     def device(self) -> torch.device:
#         """
#         Return the device of the model.
#         """
#         return next(self.parameters()).device

#     def convert_to_bf16(self) -> None:
#         """
#         Convert the torso of the model to bfloat16.
#         """
#         self.blocks.apply(convert_module_to_bf16)


#     def convert_to_fp16(self) -> None:
#         """
#         Convert the torso of the model to float16.
#         """
#         self.blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self) -> None:
#         """
#         Convert the torso of the model to float32.
#         """
#         self.blocks.apply(convert_module_to_f32)

#     def initialize_weights(self) -> None:
#         # Initialize transformer layers:
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.apply(_basic_init)

#     def voxelize(self, positions):
#         """
#         根据给定的 voxel_size，将点云位置归类到各自的 voxel 中。
#         positions: N x 3 的 tensor
#         """
#         voxel_indices = (positions / self.voxel_size).floor().long()
#         return voxel_indices

#     # def create_patches(self, voxel_indices):
#     #     """
#     #     根据 voxel indices 划分多个 patch，每个 patch 包含若干个 voxel。
#     #     voxel_indices: N x 3 的 tensor，表示每个点所在的 voxel 索引。
#     #     """
#     #     unique_voxels = voxel_indices.unique(dim=0)  # 获取所有独特的 voxel
#     #     patches = []

#     #     # 基于一定策略将 voxels 分为多个 patch（比如临近 voxel 为一个 patch）
#     #     # 假设每个 patch 包含若干个 voxel，这里简单按 voxel 的顺序划分
#     #     num_patches = len(unique_voxels) // self.patch_size
#     #     for i in range(num_patches):
#     #         patch_voxels = unique_voxels[i*self.patch_size:(i+1)*self.patch_size]
#     #         patches.append(patch_voxels)

#     #     return patches

#     def _create_patches(self, voxel_indices, features, patch_count=None):
#         """
#         根据 voxel indices 划分多个 patch，每个 patch 包含若干个 voxel。
#         voxel_indices: N x 3 的 tensor，表示每个点所在的 voxel 索引。
#         features: N x D 的 tensor，表示每个点的特征。
#         """
#         # 对点云位置使用 Hilbert 曲线进行排序
#         hilbert_indices = encode(voxel_indices, num_dims=3, num_bits=10)

#         # 将排序后的索引转换为 tensor，并基于 Hilbert 曲线的索引进行排序
#         hilbert_indices_tensor = torch.tensor(hilbert_indices)
#         sorted_indices = torch.argsort(hilbert_indices_tensor)

#         # 根据排序后的索引对 voxel_indices 和 features 进行排序
#         sorted_features = features[sorted_indices]

#         # 使用 torch 的切片操作一次性分割成多个 patch
#         num_points = sorted_features.size(0)

#         # 计算总的 patch 数量（保证每个patch最大包含 patch_size 个点，最后一个 patch 点数可能不等）
#         patches = []
#         patch_count = patch_count if patch_count else (seqlen + self.patch_size - 1) // self.patch_size  # 计算总的 patch 数量

#         # 提取前 N-1 个 patch，这些 patch 大小是固定的 patch_size
#         patches.append(sorted_features[:(patch_count - 1) * self.patch_size].view(
#             1, patch_count - 1, self.patch_size, self.model_channels
#         ))
#         # 处理最后一个 patch（大小不等）
#         last_patch_size = num_points - (patch_count - 1) * self.patch_size
#         patches.append(sorted_features[(patch_count - 1) * self.patch_size:].view(
#            1, last_patch_size, self.model_channels
#         ))

#         # 生成逆索引来恢复原始顺序
#         inverse_indices = torch.argsort(sorted_indices)

#         return patches, sorted_indices, inverse_indices

#     def _process_attention_blocks(self, tokens, seq_len, patch_count, block_idx, blocks, block_type, pos=None):
#         """Process attention blocks with tokens in shape (B*S, P, C)."""

#         if block_type == 'patch':  # local
#             # 对于 patch，前 N-1 个合并成 [1, (P-1), patch_size, N， D]
#             B, prior_patch_count = tokens[0].shape[:2]
#             remain_patch_size = tokens[-1].shape[1]
#             # 合并前 N-1 个 patch (每个 patch 的形状是 [patch_size, D])
#             local_tokens = torch.cat([t for t in tokens[:-1]], dim=1)  # 组合前 N-1 个 patch
#             local_tokens = local_tokens.view(B*(patch_count - 1), self.patch_size, self.model_channels)
#             print(pos.shape, pos)
#             local_pos = pos[:(patch_count - 1)*self.patch_size].view(B*(patch_count - 1), self.patch_size, 3) if pos is not None else None
#             print(f'local_tokens: {local_tokens.shape}, local_pos: {local_pos.shape}')
#             # 处理最后一个 patch，形状是 [real_patch_size, D]
#             last_patch_tokens = tokens[-1].view(B, remain_patch_size, self.model_channels)

#             last_patch_pos = pos[:-remain_patch_size].view(remain_patch_size, 3) if pos is not None else None

#             # 在训练模式下使用 checkpoint 进行内存优化
#             local_tokens = blocks[block_idx](local_tokens, pos=local_pos)
#             last_patch_tokens = blocks[block_idx](last_patch_tokens, pos=last_patch_pos)

#         elif block_type == 'global':  # global

#             target_shape = (b, seq_len * patch_count, embed_dim)
#             pos_target_shape = (b, seq_len * patch_count, 2) if pos is not None else None



#         # 恢复 tokens 的原始形状
#         return tokens.reshape(*token_shape)

#     def forward(self, features, positions, voxel_points=None):
#         print(f'000 features: {features.shape}')
#         features = self.input_layer(features) # [N, 768]
#         print(f'111 features: {features.shape}')

#         positions = positions.float()
#         if self.pe_mode == "ape":
#             features = features + self.pos_embedder(positions)
#         elif self.pe_mode == "ptv3pe":
#             features = features + self.pos_embedder(positions)
#         elif self.pe_mode == "dual":
#             features = features + self.pos_embedder(voxel_points, positions)
#         elif self.pe_mode == "ptv3rope":
#             features = features + self.pos_embedder(positions)
#         elif self.pe_mode =='rope':
#             pass
#         else:
#             raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")
#         print(f'222 features: {features.shape}')

#         # check_mem(msg='[TransformerBase] pe')
#         seqlen =  features.shape[0]
#         # features = features.reshape(seqlen, self.num_heads, self.num_head_channels) # [S, N, D]
#         voxel_indices = self.voxelize(positions)
#         patch_count = (seqlen + self.patch_size - 1) // self.patch_size  # 计算总的 patch 数量
#         features, sorted_indices, inverse_indices = self._create_patches(voxel_indices, features, patch_count)
#         print([_.shape for _ in features])
#         voxel_indices = voxel_indices[sorted_indices]
#         positions = positions[sorted_indices]

#         outputs = []

#         # Loop through the attention depth
#         for idx in range(self.num_blocks):
#             # Process local tokens with frame blocks (local attention)
#             features = self._process_attention_blocks(
#                 tokens=features, #if global_tokens is None else global_tokens,
#                 seq_len=seqlen,
#                 patch_count=patch_count,
#                 block_idx=idx,
#                 blocks=self.patch_blocks,
#                 block_type='patch',
#                 pos=positions,
#             )  # [1, V, cond+L, C]

#             # Process global tokens with global blocks (global attention)
#             features = self._process_attention_blocks(
#                 tokens=features,
#                 b=b,
#                 seq_len=seq_len,
#                 patch_count=patch_count,
#                 embed_dim=embed_dim,
#                 block_idx=idx,
#                 blocks=self.global_blocks,
#                 block_type='global',
#                 pos=positions, #
#             )  # [1, V, cond+L, C]

#             # Combine frame and global intermediates
#             if idx in self.intermediate_idxs:  # e.g., [4, 11, 17, 23]
#                 combined_out = torch.cat([local_tokens, global_tokens], dim=-1)  # [B, V, cond+L, 1024+1024]
#                 outputs.append(combined_out)

#         return outputs, inverse_indices

# if __name__ == '__main__':
#     attn = PatchAttention(
#         in_channels=640,
#         model_channels=640,
#         num_blocks=1,
#         num_heads=10,
#         attn_mode="full",
#         pe_mode="ptv3pe",
#         use_fp16=False,
#         use_bf16=True,
#         use_checkpoint=True,
#         qk_rms_norm=False,
#         backend='flash_attn_npu',
#         voxel_size = 0.001,
#         patch_size = 64
#     )

#     N = 200_000
#     C = 640
#     features = torch.rand((N, C), dtype=torch.bfloat16, device='cuda:1')
#     positions = torch.rand((N, C), dtype=torch.bfloat16, device='cuda:1')

#     out = attn(features, positions)
#     print([_.shape for _ in out])

    