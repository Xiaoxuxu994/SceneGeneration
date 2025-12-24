import torch
from src.models.GSdecoder.base import PatchAttention

# 设置模型参数
N = 5045         # 点云中点的数量
D = 640          # 每个点的特征维度
M = 3            # 点云的维度（3D）
patch_size = 64  # 每个patch的大小
depth = 4        # 模型的深度（注意力层数）
voxel_size = 0.001 # Voxel大小（假设值）
num_heads = 10    # 每层多头注意力的头数
num_head_channels = 64

device = 'npu:0'

# 随机生成点云数据
features = torch.randn(N, D).to(device)  # N个点，每个点有D维特征
positions = torch.randn(N, M).to(device) # N个点，每个点有M维位置（3D）

# 假设我们使用的位置编码
pos_emb = torch.randn(N, M).to(device)

# 创建 PatchAttention 模型实例
model = PatchAttention(
    in_channels=D,
    model_channels=D,
    num_blocks=depth,
    num_heads=num_heads,
    mlp_ratio=4.0,
    attn_mode="full",
    window_size=None,
    pe_mode="ape",
    use_fp16=False,
    use_bf16=False,
    use_checkpoint=False,
    qk_rms_norm=False,
    dropout=0.1,
    backend="flash_attn_npu",
    voxel_size=voxel_size,
    patch_size=patch_size
).to(device)

# 执行 forward 方法
outputs, inverse_indices = model(features, positions)

# 打印输出的形状以验证
for output in outputs:
    print(f"Output shape: {output.shape}")
print(f"Inverse indices shape: {inverse_indices.shape}")