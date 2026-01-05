import torch
import torch.nn as nn

from transformers import AutoModel
from einops import rearrange
from .blocks.utils import init_weights

from submodules.pi3.models.dinov2.hub.backbones import dinov2_vitl14_reg, dinov2_vits14_reg

from .blocks import (
    LinearTokenizer, 
    GlobalxLocalSelfAttnBlock, 
    DPTHead, 
    init_weights, 
    LinearUntokenizer, 
)


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class DepthAligner(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        
        self._init_tokenizers()
        
        input_token_dim = 0
        if cfg.tokenizer.image_tokenizer.enabled:
            input_token_dim += cfg.tokenizer.image_tokenizer.d_token
        self.token_dim_mapping = nn.Sequential(
            nn.LayerNorm(input_token_dim), 
            nn.Linear(input_token_dim, cfg.d_model, bias=False), #.apply(init_weights), 
        )
        
        if cfg.depth_scaler.enabled:
            self.depth_scale_token = torch.nn.Parameter(data=torch.randn((cfg.d_model)), requires_grad=True)
            self.depth_scale_decoder = nn.Sequential(
                nn.LayerNorm(cfg.d_model), 
                nn.Linear(cfg.d_model, 2, bias=False), 
            ).apply(init_weights)
            self.depth_scale_activater = lambda x: torch.exp(x)
            self.depth_trans_activater = lambda x: x
        
        assert cfg.transformer_layer_count % 2 == 0
        self.blocks = nn.ModuleList(
            [GlobalxLocalSelfAttnBlock(dim=cfg.d_model, head_dim=cfg.d_model//cfg.n_heads) for _ in range(cfg.transformer_layer_count)]
        )
        
        self.depth_head = self._create_head(self.cfg.de_tokenizer.type, 1, activation="exp")

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN), 
            ("_resnet_std", _RESNET_STD), 
        ):
            self.register_buffer(
                name, 
                torch.FloatTensor(value).view(1, 1, 3, 1, 1), 
                persistent=False, 
            )
    
    def _create_head(self, head_type, d_out, activation):
        if head_type == "dpt":
            return DPTHead(self.cfg.d_model, output_dim=d_out+1, intermediate_layer_idx=[3, 11, 17, 23], 
                           activation=activation, conf_activation="2sigmoid")
        elif head_type == "linear":
            return LinearUntokenizer(self.patch_size, self.cfg.d_model, d_out, activation)
        raise NotImplementedError(f"Detokenizer type {head_type} is not implemented yet.")
    
    def _build_tokenizer(self, cfg):
        if not cfg.enabled:
            return None
        if cfg.type.startswith("dinov2"):
            if cfg.type.endswith("large"):
                tokenizer = dinov2_vitl14_reg(pretrained=True, model_dir=cfg.ckpt_path)
            elif cfg.type.endswith("small"):
                tokenizer = dinov2_vits14_reg(pretrained=True, model_dir=cfg.ckpt_path)
            else:
                raise NotImplementedError("Tokenizer type '{}' is not implemented.".format(cfg.type))
            del tokenizer.mask_token
            return tokenizer
        elif cfg.type == "linear":
            return LinearTokenizer(cfg.d_in, cfg.d_token, cfg.patch_size)
        elif cfg.type == "identity":
            return nn.Identity()
        else:
            raise NotImplementedError("Tokenizer type '{}' is not implemented.".format(cfg.type))
    
    def _init_tokenizers(self):
        if self.cfg.tokenizer.image_tokenizer.enabled:
            self.image_tokenizer  = self._build_tokenizer(self.cfg.tokenizer.image_tokenizer)
    
    def embed_input_to_tokens(self, images: torch.Tensor):
        
        B, V, C, H, W = images.shape
        token_list = []
        
        if getattr(self, "image_tokenizer", None) is not None:
            dino_output  = self.image_tokenizer(((images - self._resnet_mean) / self._resnet_std).reshape(B * V, C, H, W), is_training=True)
            image_tokens = dino_output["x_norm_patchtokens"]
            image_tokens = rearrange(image_tokens, "(b v) l d -> b (v l) d", v=V)
            token_list.append(image_tokens)
        
        input_tokens = torch.cat(token_list, dim=-1)
        tokens = input_tokens.reshape(B, -1, input_tokens.shape[-1])
        
        tokens = self.token_dim_mapping(tokens) # b (v l) d
        
        if getattr(self, "depth_scale_token", None) is not None:
            tokens = rearrange(tokens, "b (v l) d -> b v l d", v=V)
            expanded_depth_scale_token = self.depth_scale_token[None, None, None].expand(B, V, -1, -1) # b v 1 d
            tokens = torch.cat([tokens, expanded_depth_scale_token], dim=-2) # b v (l+1) d
            tokens = rearrange(tokens, "b v l d -> b (v l) d")
        
        return tokens
    
    def decompose_depth_scale_token(self, tokens: torch.Tensor):
        """Decompose depth scale token from the scale token concatenated token.

        Args:
            tokens (torch.Tensor): The tokens to be decomposed, should be of shape (..., l + 1, d).
        """
        if self.cfg.depth_scaler.enabled == True:
            return tokens[..., :-1, :], tokens[..., -1:, :]
        return tokens, None
    
    
    def forward_camera_tokens(self, images: torch.Tensor, pe_helper_global=None, pe_helper_local=None):
        """_summary_

        Args:
            images (torch.Tensor): Input images, tensor of shape (B, V, 3, H, W).
            pe_helper_global (AtomicAttnPEHelper, optional): Positional embedding helper for global frame attention. Defaults to None.
            pe_helper_local (AtomicAttnPEHelper, optional): Positional embedding helper for local frame attention. Defaults to None.
        """
        B, V, C, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        assert ph * self.patch_size == H and pw * self.patch_size == W
        
        tokens = self.embed_input_to_tokens(images)
        
        token_list = []
        num_blocks = len(self.blocks)
        for block in self.blocks[:num_blocks//2]:
            block_token_list = block(tokens, V, pe_helper_global, pe_helper_local)
            token_list.extend([self.decompose_depth_scale_token(
                rearrange(token, "b (v l) d -> b v l d", v=V))[0] for token in block_token_list])
            tokens = block_token_list[-1]
        
        camera_token = tokens.clone() # rearrange(tokens, "b v l d -> b (v l) d")
        
        return token_list, camera_token
    
    def forward_depth_tokens(self, images: torch.Tensor, token_list: list[torch.Tensor], pe_helper_global=None, pe_helper_local=None):
        """_summary_

        Args:
            camera_token_list (_type_): _description_
            pe_helper_global (_type_, optional): _description_. Defaults to None.
            pe_helper_local (_type_, optional): _description_. Defaults to None.
        """
        B, V, C, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        assert ph * self.patch_size == H and pw * self.patch_size == W
        
        num_blocks = len(self.blocks)
        tokens = rearrange(token_list[-1], "b v l d -> b (v l) d")
        for block in self.blocks[num_blocks//2:]:
            block_token_list = block(tokens, V, pe_helper_global, pe_helper_local)
            token_list.extend([self.decompose_depth_scale_token(
                rearrange(token, "b (v l) d -> b v l d", v=V))[0] for token in block_token_list])
            tokens = block_token_list[-1]
        
        for idx in range(len(token_list)):
            token_list[idx] = token_list[idx].reshape(B, V, -1, token_list[idx].shape[-1])
        if self.cfg.de_tokenizer.type == "dpt":
            aligned_depths, aligned_depths_conf = self.depth_head(token_list, images, 0)
        elif self.cfg.de_tokenizer.type == "linear":
            aligned_depths, aligned_depths_conf = self.depth_head(token_list[-1], (H, W))
        
        depth_scales, depth_transs = None, None
        if getattr(self, "depth_scale_token", None) is not None:
            updated_depth_scale_tokens = self.decompose_depth_scale_token(rearrange(block_token_list[-1], "b (v l) d -> b v l d", v=V))[1] # b v 1 d
            depth_scale_factors = self.depth_scale_decoder(updated_depth_scale_tokens)
            depth_scales = self.depth_scale_activater(depth_scale_factors[..., :1].squeeze()) # b v
            depth_transs = self.depth_trans_activater(depth_scale_factors[..., 1:].squeeze()) # b v
            scaled_depths = aligned_depths.detach() * depth_scales[..., None, None, None] + depth_transs[..., None, None, None] # b v 1 h w
        else:
            scaled_depths = aligned_depths
                
        return {
            "aligned_depths":      aligned_depths, 
            "aligned_depths_conf": aligned_depths_conf.reshape(*aligned_depths.shape), 
            "scaled_depths":       scaled_depths, 
            "depth_scales":        depth_scales, 
            "depth_transs":        depth_transs, 
        }
    
    def forward(self, images: torch.Tensor, pe_helper_global=None, pe_helper_local=None):
        """Depth aligner forward pass. All inputs are resized before passing into the module.

        Args:
            images (torch.Tensor): Input image of shape (B, V, 3, H, W)
        
        Returns:
            A dictionary containing results.
        """
        
        B, V, C, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        assert ph * self.patch_size == H and pw * self.patch_size == W
        
        tokens = self.embed_input_to_tokens(images)
        
        token_list = []
        for block in self.blocks:
            block_token_list = block(tokens, V, pe_helper_global, pe_helper_local)
            token_list.extend([self.decompose_depth_scale_token(
                rearrange(token, "b (v l) d -> b v l d", v=V))[0] for token in block_token_list])
            tokens = block_token_list[-1]
        
        aligned_depths_tokens = [rearrange(token, "b v l d -> b (v l) d") for idx, token in enumerate(token_list) if idx % 2 == 1]
        
        for idx in range(len(token_list)):
            token_list[idx] = token_list[idx].reshape(B, V, -1, token_list[idx].shape[-1])
        if self.cfg.de_tokenizer.type == "dpt":
            aligned_depths, aligned_depths_conf = self.depth_head(token_list, images, 0)
        elif self.cfg.de_tokenizer.type == "linear":
            aligned_depths, aligned_depths_conf = self.depth_head(token_list[-1], (H, W))
        
        depth_scales, depth_transs = None, None
        if getattr(self, "depth_scale_token", None) is not None:
            updated_depth_scale_tokens = self.decompose_depth_scale_token(rearrange(block_token_list[-1], "b (v l) d -> b v l d", v=V))[1] # b v 1 d
            depth_scale_factors = self.depth_scale_decoder(updated_depth_scale_tokens)
            depth_scales = self.depth_scale_activater(depth_scale_factors[..., :1].squeeze()) # b v
            depth_transs = self.depth_trans_activater(depth_scale_factors[..., 1:].squeeze()) # b v
            scaled_depths = aligned_depths.detach() * depth_scales[..., None, None, None] + depth_transs[..., None, None, None] # b v 1 h w
        else:
            scaled_depths = aligned_depths
        
        return {
            "aligned_depths":        aligned_depths, 
            "aligned_depths_conf":   aligned_depths_conf.reshape(*aligned_depths.shape), 
            "aligned_depths_tokens": aligned_depths_tokens, 
            "scaled_depths":         scaled_depths, 
            "depth_scales":          depth_scales, 
            "depth_transs":          depth_transs, 
        }
