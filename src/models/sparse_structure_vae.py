from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")

# 计算填充的函数
def calculate_padding(input_size, kernel_size, stride, dilation=1):
    output_size = (input_size + 2 * dilation - kernel_size) // stride + 1
    pad_needed = max(0, (output_size - 1) * stride + kernel_size - input_size)
    padding_left = pad_needed // 2
    padding_right = pad_needed - padding_left
    return padding_left, padding_right

class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h

class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":            
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):        
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)

class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels*8, kernel_size=3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if hasattr(self, "conv"):
        if self.mode == "conv":
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        


class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).
    
    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        print("***************************** SparseStructureEncoder ******************************")
        print(f"""
              in_channels: {in_channels}, latent_channels: {latent_channels}, num_res_blocks: {num_res_blocks}, 
              channels: {channels}, num_res_blocks_middle: {num_res_blocks_middle}, 
              norm_type: {norm_type}, use_fp16: {use_fp16}
              """)
        print("************************************************************************************")
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()
        
        self.required = int(2*(len(channels)-1))
    
    def pad_input(self, tensor):
        # 获取当前 tensor 的形状
        shape = tensor.shape
        # 初始化填充量
        padding = [0,0,0,0]

        # 遍历每个维度，计算填充量
        for dim_size in shape[2:]:
            remainder = dim_size % self.required
            if remainder == 0:
                padding.extend([0,0])  # 如果已经是 4 的倍数，不需要填充
            else:
                # 计算填充量，使其变为 4 的倍数
                pad_needed = self.required - remainder
                padding.append(pad_needed // 2)  # 前面填充 pad_needed // 2
                padding.append(pad_needed - pad_needed // 2)  # 后面填充剩余的部分

        # 由于 F.pad 需要按 (last_dim, second_last_dim, ..., first_dim) 的顺序传递填充量
        padding = padding[::-1]  # 翻转填充顺序
        
        # 使用 F.pad 进行填充
        return F.pad(tensor, padding)
        

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        
        h = self.input_layer(x) # ([1, 1, 68, 36, 26]) -> ([1, 32, 68, 36, 26])
        h = h.type(self.dtype) 
        
        for block in self.blocks:
            h = block(h) # ([1, 32, 68, 36, 26]), ([1, 32, 68, 36, 26]), ([1, 128, 34, 18, 13]), ([1, 128, 34, 18, 13]), ([1, 128, 34, 18, 13]), ([1, 512, 17, 9, 6]), ([1, 512, 17, 9, 6]),  ([1, 512, 17, 9, 6])
        h = self.middle_block(h) # ([1, 512, 17, 9, 6])

        h = h.type(x.dtype)
        h = self.out_layer(h) # ([1, 16, 17, 9, 6])


        mean, logvar = h.chunk(2, dim=1) # (torch.Size([1, 8, 17, 9, 6]), torch.Size([1, 8, 17, 9, 6]))

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
            
        if return_raw:
            return z, mean, logvar
        return z # (torch.Size([1, 8, 17, 9, 6])
        

class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).
    
    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """ 
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x) # ([1, 8, 17, 9, 6]) -> ([1, 32, 17, 9, 6])
        
        h = h.type(self.dtype)
                
        h = self.middle_block(h) # ([1, 32, 17, 9, 6])
        for block in self.blocks:
            h = block(h) # [([1, 32, 17, 9, 6]), ([1, 32, 17, 9, 6]), ([1, 128, 34, 18, 12]), ([1, 128, 34, 18, 12]), ([1, 128, 34, 18, 12]), ([1, 512, 68, 36, 24]), ([1, 512, 68, 36, 24]), ([1, 512, 68, 36, 24])]

        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h


# from ...registry import MODELS

# @MODELS.register_module()
class SparseStructureVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_channels=8, num_res_blocks=2, 
                 channels=[32, 128, 512], use_fp16=False):
        super().__init__()
        # encode 默认output channel是latent_channels*2
        self.model_encoder = SparseStructureEncoder(in_channels=in_channels,
                                                    latent_channels=latent_channels,
                                                    num_res_blocks=num_res_blocks,
                                                    channels=channels,
                                                    use_fp16=use_fp16)
        self.model_decoder = SparseStructureDecoder(out_channels=out_channels,
                                                    latent_channels=latent_channels,
                                                    num_res_blocks=num_res_blocks,
                                                    channels=channels,
                                                    use_fp16=use_fp16)
        self.required = int(2*(len(channels)-1))
    
    def pad_input(self, tensor):
        # 获取当前 tensor 的形状
        shape = tensor.shape
        # 初始化填充量
        padding = [0,0,0,0]

        # 遍历每个维度，计算填充量
        for dim_size in shape[2:]:
            remainder = dim_size % self.required
            if remainder == 0:
                padding.extend([0,0])  # 如果已经是 4 的倍数，不需要填充
            else:
                # 计算填充量，使其变为 4 的倍数
                pad_needed = self.required - remainder
                padding.append(pad_needed // 2)  # 前面填充 pad_needed // 2
                padding.append(pad_needed - pad_needed // 2)  # 后面填充剩余的部分

        # 由于 F.pad 需要按 (last_dim, second_last_dim, ..., first_dim) 的顺序传递填充量
        padding = padding[::-1]  # 翻转填充顺序
        
        # 使用 F.pad 进行填充
        return F.pad(tensor, padding)
        
    def encode(self, x, sample_posterior=False):
        """
        arg:
            x : Binary grid tensor [B, in_channels, num_gridX, num_gridY, num_gridZ]
        return:
            ss_latent: [B, latent_channels, downsampleX, downsampleY, downsampleZ] 
        """
        ss_latent = self.model_encoder(x, sample_posterior=sample_posterior)
        return ss_latent

    def decode(self, ss_latent):
        voxels = self.model_decoder(ss_latent)
        return voxels

    def forward(self, x, sample_posterior=False):
        pad_x = self.pad_input(x)
        ss_latent = self.model_encoder(pad_x, sample_posterior=sample_posterior)
        voxels = self.model_decoder(ss_latent)
        # print(f'x.shape: {x.shape}, pad_x.shape:{pad_x.shape}, ss_latent.shape: {ss_latent.shape}, voxels.shape: {voxels.shape}')
        return voxels, pad_x