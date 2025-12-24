from typing import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from ...modules.utils import convert_module_to_f16, convert_module_to_bf16, convert_module_to_f32
from ...modules import sparse as sp
from ...modules.attention import RotaryPositionEmbedder
from ...modules.transformer import AbsolutePositionEmbedder, PTV3PosEmbedding, DualPathPositionalEncoding, PTV3RopeEmbedding
from ...modules.transformer.blocks import TransformerBlock, TransformerCrossBlock
from ...modules.sparse.transformer import SparseTransformerBlock, SparseTransformerCrossBlock


def check_mem(msg=''):
    # 查看当前NPU显存已分配的大小（以字节为单位）
    allocated_memory = torch.npu.memory_allocated()
    # 查看当前NPU显存的已保留部分（以字节为单位）
    reserved_memory = torch.npu.memory_reserved()
    print(f"{msg} Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB, Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")


def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope", "ptv3pe", "dual"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
        isPTV3PELog: bool = False,
        backend: str = 'flash_attn_npu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        self.dropout = dropout

        if self.pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif self.pe_mode == "ptv3pe":
            self.pos_embedder = PTV3PosEmbedding(hidden_dim=192, dim=model_channels, isLog=isPTV3PELog)
        elif self.pe_mode == "dual":
            self.pos_embedder = DualPathPositionalEncoding(max_voxel_dim=128,
                                                        index_embed_dim=128,      # per axis
                                                        coord_embed_dim=model_channels-128*3, # 128*3 + 640 = 1024
                                                        coord_type='fourier',     # or 'mlp'
                                                        num_freqs=16              # optional
                                                    )
        elif self.pe_mode == 'rope':
            pass
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                dropout=self.dropout,
                # backend=backend,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, sp.SparseLinear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, voxel_points=None) -> sp.SparseTensor:
        """
        voxel_points: [N, 3] float coordinates
        """
        h = self.input_layer(x) # [N, 768]
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        elif self.pe_mode == "ptv3pe":
            h = h + self.pos_embedder(x.coords[:, 1:].float())
        elif self.pe_mode == "dual":
            h = h + self.pos_embedder(voxel_points, x.coords[:, 1:])
        elif self.pe_mode == 'rope':
            pass
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")
        h = h.type(self.dtype) # [N, 768]
        for block in self.blocks:
            h = block(h)
        return h
    
# === Sinusoidal 3D Positional Encoding ===
class SinusoidalPE3D(nn.Module):
    def __init__(self, num_freqs=10, out_dim=1024):
        super().__init__()
        self.num_freqs = num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(3 * num_freqs * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, xyz_inds):  # [N, 3]
        freqs = 2 ** torch.arange(self.num_freqs, device=xyz_inds.device).float()
        xyz = xyz_inds.unsqueeze(-1) * freqs  # [N, 3, F]
        pe = torch.cat([torch.sin(xyz), torch.cos(xyz)], dim=-1)  # [N, 3, 2F]
        return self.mlp(pe.view(xyz_inds.shape[0], -1))  # [N, out_dim]

# Fourier PE（PositionalEncoding）模块
class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords, rays=None):
        batch_size, dim = coords.shape
        coords = coords.unsqueeze(-1)  # [N, D, 1]

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves).float().to(coords.device)
        multipliers = 2 ** octaves * math.pi  # [num_octaves]
        for _ in range(len(coords.shape) - 2):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers  # [N, D, num_octaves]
        sines = torch.sin(scaled_coords).reshape(batch_size, -1)
        cosines = torch.cos(scaled_coords).reshape(batch_size, -1)
        return torch.cat([sines, cosines], dim=-1)  # [N, D * num_octaves * 2]

class TransformerBase(nn.Module):
    """
    Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope", "ptv3pe", "ptv3rope" "dual"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
        backend: str = 'flash_attn_npu',        
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.backend = backend
            
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif self.pe_mode == "ptv3pe":
            self.pos_embedder = PTV3PosEmbedding(hidden_dim=192, dim=model_channels)
        elif self.pe_mode == "dual":
            self.pos_embedder = DualPathPositionalEncoding(max_voxel_dim=128,
                                                        index_embed_dim=128,      # per axis
                                                        coord_embed_dim=model_channels-128*3, # 128*3 + 640 = 1024
                                                        coord_type='fourier',     # or 'mlp'
                                                        num_freqs=16              # optional
                                                    )
        elif self.pe_mode == "ptv3rope":
            self.pos_embedder = PTV3RopeEmbedding(hidden_dim=192, dim=model_channels,
                                                  act_layer=nn.LeakyReLU)
        elif self.pe_mode == 'rope':
            pass
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")

        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            

        self.input_layer = nn.Linear(in_channels, model_channels)
        # if pe_mode == "ape":
        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_window=shift_window,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                backend=backend,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])
    
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, voxel_points=None) -> torch.Tensor:
        # check_mem(msg='[TransformerBase] start forward')
        if 'nsa_npu_sample'==self.backend: kv_index = torch.linspace(0, x.shape[0]-1, 14000, device=x.device).int() # 在NPU上生成索引
        h = self.input_layer(x) # [N, 768]
        coords = coords.float()
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(coords)
        elif self.pe_mode == "ptv3pe":
            h = h + self.pos_embedder(coords)
        elif self.pe_mode == "dual":
            h = h + self.pos_embedder(voxel_points, coords)
        elif self.pe_mode == "ptv3rope":
            h = h + self.pos_embedder(coords)
        elif self.pe_mode =='rope':
            pass
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")
        # check_mem(msg='[TransformerBase] pe')        
        h = h.to(self.dtype) # h = h.type(self.dtype) # [N, 768]
        if 'nsa_npu_sample' == self.backend:
            for block in self.blocks:
                h = block(h, kv_index)
        else:
            for block in self.blocks:
                h = block(h)
            # check_mem(msg='[TransformerBase] attn')        
        return h.squeeze(0)

class TransformerBaseWithCatPE(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope", "sin3d", "fourier"] = "ape",
        pe_dim: int = 256,
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        self.dropout = dropout

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(pe_dim)
        # elif pe_mode == "sin3d":
        #     self.pos_embedder = SinusoidalPE3D(num_freqs=100, out_dim=pe_dim)
        #     self.input_layer = nn.Linear(in_channels, model_channels - pe_dim)
        # elif pe_mode == "fourier":
        #     self.pos_embedder = PositionalEncoding(num_octaves=8, start_octave=0)
        #     self.input_layer = nn.Linear(in_channels, model_channels - 2 * (8 * 3))  # 8 octaves, 3D coords
        elif self.pe_mode == "ptv3pe":
            self.pos_embedder = PTV3PosEmbedding(hidden_dim=192, dim=pe_dim)
        elif self.pe_mode == "dual":
            self.pos_embedder = DualPathPositionalEncoding(max_voxel_dim=128,
                                                           index_embed_dim=128,      # per axis
                                                           coord_embed_dim=model_channels-128*3, # 128*3 + 640 = 1024
                                                           coord_type='fourier',     # or 'mlp'
                                                           num_freqs=16              # optional
                                                    )
        else:
            raise NotImplementedError
        
        self.input_layer = nn.Linear(in_channels, model_channels)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_window=shift_window,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x, pos) -> sp.SparseTensor:
        h = self.input_layer(x) # [N, 768]
        
        if self.pe_mode == "ape":
            h = torch.cat([ h , self.pos_embedder(coords) ], dim=-1)
        elif self.pe_mode == "ptv3pe":
            h = torch.cat([ h , self.pos_embedder(coords.float())], dim=-1)
        elif self.pe_mode == "dual":
            h = torch.cat([ h , self.pos_embedder(voxel_points, coords) ], dim=-1)
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")
        h = h.to(self.dtype) # h = h.type(self.dtype) # [N, 768]
        for block in self.blocks:
            h = block(h)
        return h.squeeze(0)
    

class TransformerCrossBase(nn.Module):
    """
    Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        ctx_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.ctx_channels = ctx_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        self.dropout = dropout

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif self.pe_mode == "ptv3pe":
            self.pos_embedder = PTV3PosEmbedding(hidden_dim=192, dim=model_channels)
        elif self.pe_mode == "dual":
            self.pos_embedder = DualPathPositionalEncoding(max_voxel_dim=128,
                                                        index_embed_dim=128,      # per axis
                                                        coord_embed_dim=model_channels-128*3, # 128*3 + 640 = 1024
                                                        coord_type='fourier',     # or 'mlp'
                                                        num_freqs=16              # optional
                                                    )            
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            TransformerCrossBlock(
                model_channels,
                ctx_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_window=shift_window,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                # dropout=self.dropout,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor, cxt: torch.Tensor, coords: torch.Tensor = None, coords_cxt: torch.Tensor = None, 
                return_scores=False, voxel_points=None, voxel_points_ctx=None) -> torch.Tensor:
        h = self.input_layer(x) # [N, 768]
        cxt = self.input_layer(cxt)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(coords) if coords is not None else h
            cxt = cxt + self.pos_embedder(coords_cxt) if coords_cxt is not None else cxt
        elif self.pe_mode == "ptv3pe":
            h = h + self.pos_embedder(coords.float()) if coords is not None else h
            cxt = cxt + self.pos_embedder(coords_cxt.float()) if coords_cxt is not None else cxt
        elif self.pe_mode == "dual":
            h = h + self.pos_embedder(voxel_points, coords) if coords is not None else h
            cxt = cxt + self.pos_embedder(voxel_points_ctx, coords_cxt) if coords_cxt is not None else cxt
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")
        
        h = h.type(self.dtype) # [N, 768]
        cxt = cxt.type(self.dtype) # [N, 768]
        for block in self.blocks:
            if return_scores: 
                h, weights = block(h, cxt, return_scores)
                return h, weights
            else:
                h = block(h, cxt)
        return h.squeeze(0)

class Upsampler(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope", "sin3d", "fourier", "fourier+learnable"] = "ape",
        pe_dim: int = 256,
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.fu = upsample_ratio
        self.dropout = dropout

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(pe_dim)
            self.input_layer = nn.Linear(in_channels, model_channels - pe_dim)
        elif pe_mode == "sin3d":
            self.pos_embedder = SinusoidalPE3D(num_freqs=100, out_dim=pe_dim)
            self.input_layer = nn.Linear(in_channels, model_channels - pe_dim)
        elif pe_mode == "fourier":
            self.pos_embedder = PositionalEncoding(num_octaves=42, start_octave=0)
            self.input_layer = nn.Linear(in_channels, model_channels - 2 * (42 * 3))  # 8 octaves, 3D coords
        elif pe_mode == "learnable":
            self.pos_embedder = nn.Parameter(torch.randn(upsample_ratio, pe_dim))
            self.input_layer = nn.Linear(in_channels, model_channels - pe_dim)
        elif pe_mode == "fourier+learnable":
            self.learnable_pe = nn.Parameter(torch.randn(upsample_ratio, pe_dim))
            self.pos_embedder = PositionalEncoding(num_octaves=42, start_octave=0)
            self.input_layer = nn.Linear(in_channels, model_channels - 2 * (42 * 3) - pe_dim)
        else:
            raise NotImplementedError
        

        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_window=shift_window,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, z, voxel_xyz) -> torch.Tensor:
        """
        z: [N, C] feature 
        voxel_xyz: [N, 3] voxel center
        """
        N, C = z.shape
        z_expand = z.unsqueeze(1).expand(N, self.fu, C)  # [N, fu, C]
        if self.pe_mode == "fourier+learnable":
            pe_learn = self.learnable_pe.unsqueeze(0).expand(N, -1, -1)  # [N, fu, C_pe]
            pe_embed = self.pos_embedder(pos) # [N, C_pe]
            pe_embed = pe_embed.unsqueeze(1).expand(N, self.fu, -1) # [N, fu, C_pe]
            pe_all = torch.cat([pe_embed, pe_learn], dim=-1)
        else:
            pe_embed = self.pos_embedder(pos) # [N, fu, C_pe]
            pe_all = torch.cat([pe_embed, pe_learn], dim=-1)

        for block in self.blocks:
            h = block(h)
        return h

class SparseTransformerCrossBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        ctx_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
        isPTV3PELog: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.ctx_channels = ctx_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        self.dropout = dropout

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        elif self.pe_mode == "ptv3pe":
            self.pos_embedder = PTV3PosEmbedding(hidden_dim=192, dim=model_channels, isLog=isPTV3PELog)  
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerCrossBlock(
                model_channels,
                ctx_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                dropout=self.dropout,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, cxt: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x) # [N, 768]
        cxt = self.input_layer(cxt)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
            cxt = cxt + self.pos_embedder(cxt.coords[:, 1:])
        elif self.pe_mode == "ptv3pe":
            h = h + self.pos_embedder(x.coords[:, 1:].float())
            cxt = cxt + self.pos_embedder(cxt.coords[:, 1:].float())        
        else:
            raise NotImplementedError(f"Unsupported positional encoding mode: {self.pe_mode}")
            
        h = h.type(self.dtype) # [N, 768]
        cxt = cxt.type(self.dtype) # [N, 768]
        for block in self.blocks:
            h = block(h, cxt)
        return h



class PosEmbed(nn.Module):
    """
    pos embedding
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        use_bf16: bool = True,
        use_fp16: bool = False,
        pe_mode: str = "ape",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(in_channels)
            
        

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, sp.SparseLinear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.pos_embedder(x.coords[:, 1:])
        h = self.input_layer(h + x) # [N, 768]            
        h = h.type(self.dtype) # [N, 768]
        return h

# class RoPE3D(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         assert dim % 2 == 0
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half = self.dim // 2
#         x1, x2 = x[..., :half], x[..., half:]

#         freq = torch.arange(half, device=device).float()
#         inv_freq = 1.0 / (10000 ** (freq / half))
#         theta = x1 * inv_freq
#         sin, cos = theta.sin(), theta.cos()
#         return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# def _init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.ones_(m.weight)
#         nn.init.zeros_(m.bias)

# # class VoxelDensityTransformer(nn.Module):
# #     def __init__(self, feat_dim=1024, pos_dim=48, embed_dim=256, nhead=8, num_layers=4):
# #         super().__init__()
# #         self.feat_proj = nn.Linear(feat_dim, embed_dim)
        
# #         self.rope = RoPE3D(pos_dim)

# #         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim+pos_dim, nhead=nhead, dim_feedforward=512, batch_first=True)
# #         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# #         self.head = nn.Sequential(
# #             nn.Linear(embed_dim+pos_dim, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, 1),
# #             nn.Tanh()  # 输出概率或密度权重
# #         )
        
        
# #         # 权重初始化
# #         self.apply(_init_weights)

# #     def forward(self, feat, pos):
# #         # feat: [N, 1024], pos: [N, 3]
# #         feat_embed = self.feat_proj(feat)  # [N, embed_dim]
# #         pos_embed = self.pos_proj(pos.float())  # [N, pos_dim]
# #         pos_rope = self.rope(pos_embed)  # [N, pos_dim]

# #         token = torch.cat([feat_embed, pos_rope], dim=-1)  # [N, embed_dim+pos_dim]

# #         token = token.unsqueeze(0)  # [1, N, D]
# #         out = self.transformer(token).squeeze(0)  # [N, D]
# #         return self.head(out).squeeze(-1)  # [N], 范围[0,1]


class VoxelDensityMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        mode: str = 'nn', # 'nn' , 'sp'
        dropout: float = 0.1,   # << New: dropout rate
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        if mode == 'sp':    
            self.mlp_layer = nn.Sequential(
                sp.SparseLinear(in_channels, in_channels//2),
                sp.SparseReLU(),
                sp.SparseLinear(in_channels//2, in_channels//4),
                sp.SparseReLU(),
                sp.SparseLinear(in_channels//4, out_channels),
                sp.SparseSigmoid(),
            )
        elif mode == 'nn':
            self.mlp_layer = nn.Sequential(
                nn.Linear(in_channels, in_channels//2),
                nn.ReLU(),
                nn.Linear(in_channels//2, in_channels//4),
                nn.ReLU(),
                nn.Linear(in_channels//4, out_channels),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError("[ERROR] mode should nn or sp !")

        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        # Initialize transformer layers:        
        for i, layer in enumerate(self.mlp_layer):
            if isinstance(layer, nn.Linear) or isinstance(layer, sp.SparseLinear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Union[sp.SparseTensor, torch.Tensor]) -> Union[sp.SparseTensor, torch.Tensor]:
        h = self.mlp_layer(x)
        return h


class VoxelDensityTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        self.dropout = dropout

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
            
        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        if pe_mode == "ape":
            self.blocks = nn.ModuleList([
                SparseTransformerBlock(
                    model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    shift_sequence=shift_sequence,
                    shift_window=shift_window,
                    serialize_mode=serialize_mode,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    qk_rms_norm=self.qk_rms_norm,
                    dropout=self.dropout,
                )
                for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
            ])
        
        self.output_layer = nn.Sequential(
            sp.SparseLinear(model_channels, model_channels//2),
            sp.SparseReLU(),
            sp.SparseLinear(model_channels//2, 1),
            sp.SparseTanh(),
        )

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        for i, layer in enumerate(self.output_layer):
            if isinstance(layer, nn.Linear) or isinstance(layer, sp.SparseLinear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x) # [N, 768]
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        # if self.pe_mode == "ptv3pe":
        #     h = torch.cat([h, self.pos_embedder(x.coords[:, 1:])], dim=-1)
        h = h.type(self.dtype) # [N, 768]
        for block in self.blocks:
            h = block(h)
        h = h.type(x.dtype) 
        h = self.output_layer(h)
        return h



class SparseTransformerLocalPEBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 256,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dropout: float = 0.1,   # << New: dropout rate
        global_pe_dim: int = 256,
        local_pe_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // 256
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        if use_bf16:
            self.dtype = torch.bfloat16
        elif use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
            
        self.dropout = dropout

        if pe_mode == "ape":
            self.global_pos_embedder = AbsolutePositionEmbedder(global_pe_dim, in_channels=3)
            self.local_pos_embedder = AbsolutePositionEmbedder(local_pe_dim, in_channels=1)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
            
        input_channels = model_channels+global_pe_dim+local_pe_dim
        self.middle_layer = sp.SparseLinear(input_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=model_channels//256,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                dropout=self.dropout,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])
        
        self.output_layer = nn.Sequential(
            sp.SparseLinear(model_channels, model_channels),
            sp.SparseReLU(),
            sp.SparseLinear(model_channels, 2*model_channels),
            sp.SparseReLU(),
            sp.SparseLinear(2*model_channels, 4*model_channels),
            sp.SparseReLU(),
            sp.SparseLinear(4*model_channels, 2*model_channels),
            sp.SparseReLU(),
            sp.SparseLinear(2*model_channels, model_channels),
            sp.SparseReLU(),
            sp.SparseLinear(model_channels, model_channels),
        )

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.blocks.apply(convert_module_to_bf16)


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        for i, layer in enumerate(self.output_layer):
            if isinstance(layer, nn.Linear) or isinstance(layer, sp.SparseLinear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, x: sp.SparseTensor, lcoalIndices: torch.Tensor) -> sp.SparseTensor:
        """
        x: feats, coords
        lcoalIndices: torch.tensor [N, 1]
        """
        # print(self.__dict__)
        # print(self.blocks)
        h = self.input_layer(x) # [N, 768]
        # print("000 h.feats: ", h.feats.shape)
        if self.pe_mode == "ape":
            h = h.replace(torch.cat((h.feats, self.global_pos_embedder(x.coords[:, 1:])), dim=-1))
            # print("111 h.feats: ", h.feats.shape)
            if lcoalIndices.shape[1] == 1:
                h = h.replace(torch.cat((h.feats, self.local_pos_embedder(lcoalIndices)), dim=-1))
        #         print("222 h.feats: ", h.feats.shape)
        # print("333 h.feats: ", h.feats.shape)
        h = self.middle_layer(h)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:])) 
        
        # h = h.type(self.dtype) # [N, 768]
        # for block in self.blocks:
        #     print("uuuuu 444 in h: ", h.feats.shape, h.feats.min(), h.feats.max())  
        #     # print(block)            
        #     h = block(h)
        #     print("uuuuu 444 out h: ", h.feats.shape, h.feats.min(), h.feats.max())     
        # h = h.type(x.dtype) 
        # h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:])) 
        
        h = self.output_layer(h)
        return h
