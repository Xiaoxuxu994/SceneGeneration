import torch
import torch.nn as nn

from .blocks import (
    init_weights, 
    DPTHead, 
    CrossAttn_Transformer_Block, 
    SelfAttn_Transformer_Block, 
    LinearUntokenizer
)


class CrossPcdPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.cam_token_proj = nn.Sequential(
            nn.LayerNorm(512), 
            nn.Linear(512, cfg.d_model, bias=False), #.apply(init_weights), 
        )
        self.cross_attn_blocks = nn.ModuleList([CrossAttn_Transformer_Block(dim=cfg.d_model, head_dim=cfg.d_model//cfg.n_heads) 
                                                for _ in range(cfg.transformer_layer_count)])
        self.self_attn_blocks  = nn.ModuleList([SelfAttn_Transformer_Block(dim=cfg.d_model, head_dim=cfg.d_model//cfg.n_heads)
                                                for _ in range(cfg.transformer_layer_count)])
        if self.cfg.de_tokenizer.depth.type == "dpt":
            assert self.cfg.transformer_layer_count == 12, "Update DPT head configuration before changing layer count."
        self.depth_head = self._create_head(self.cfg.de_tokenizer.depth.type, 1, activation="exp")
        self.image_head = self._create_head(self.cfg.de_tokenizer.image.type, 3, activation="sigmoid")
        
    def _create_head(self, head_type, d_out, activation):
        if head_type == "dpt":
            return DPTHead(self.cfg.d_model, output_dim=d_out+1, intermediate_layer_idx=[3, 5, 9, 11], activation=activation)
        elif head_type == "linear":
            return LinearUntokenizer(self.patch_size, self.cfg.d_model, d_out, activation)
        raise NotImplementedError(f"Detokenizer type {head_type} is not implemented yet.")
    
    def forward(self, cam_tokens: torch.Tensor, depth_tokens: torch.Tensor, depths: torch.Tensor, 
                pe_helper_cross_attn=None, pe_helper_self_attn=None) -> torch.Tensor:
        """Predict depth with camera tokens of group A querying depth tokens of group B.

        Args:
            cam_tokens (torch.Tensor): Camera tokens of group A, tensor of shape (b, l, d).
            depth_tokens (torch.Tensor): Depth tokens of group B, tensor of shape (b, l, d).
            depths (torch.Tensor): Depth maps, input to grasp tensor shape. 
        """
        B, V, _, H, W = depths.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        assert ph * self.patch_size == H and pw * self.patch_size == W
        cam_tokens = self.cam_token_proj(cam_tokens)
        tokens     = cam_tokens
        token_list = []
        
        if self.cfg.cross_query_type == "token_fusion":
            """Fuse target camera token with source depth token in all layers. 
            """
            for cat_block, sat_block, depth_token in zip(self.cross_attn_blocks, self.self_attn_blocks, depth_tokens):
                tokens = cat_block(tokens, depth_token, pe_helper=pe_helper_cross_attn)
                tokens = sat_block(tokens, pe_helper=pe_helper_self_attn)
                token_list.append(tokens)
        elif self.cfg.cross_query_type == "camera_fusion":
            """Directly use camera token for cross attention, and use self attention as residual operation. 
            """
            for cat_block, sat_block, depth_token in zip(self.cross_attn_blocks, self.self_attn_blocks, depth_tokens):
                tokens = cat_block(cam_tokens, depth_token, pe_helper=pe_helper_cross_attn)
                if len(token_list) == 0:
                    token_list.append(sat_block(tokens, pe_helper=pe_helper_self_attn))
                else:
                    token_list.append(sat_block(tokens + token_list[-1], pe_helper=pe_helper_self_attn))
        else:
            raise NotImplementedError(f"Cross query type {self.cfg.cross_query_type} is not implemented.")
        
        for idx in range(len(token_list)):
            token_list[idx] = token_list[idx].reshape(B, V, -1, token_list[idx].shape[-1])
        
        if self.cfg.de_tokenizer.depth.type == "dpt":
            cross_depths, cross_depths_conf = self.depth_head(token_list, depths, 0)
        elif self.cfg.de_tokenizer.depth.type == "linear":
            cross_depths, cross_depths_conf = self.depth_head(token_list[-1], (H, W))
        if self.cfg.de_tokenizer.image.type == "dpt":
            cross_images, cross_images_conf = self.image_head(token_list, depths, 0)
        elif self.cfg.de_tokenizer.image.type == "linear":
            cross_images, cross_images_conf = self.image_head(token_list[-1], (H, W))
        
        return cross_depths.reshape(B, V, 1, H, W), cross_depths_conf.reshape(B, V, 1, H, W), \
               cross_images.reshape(B, V, 3, H, W), cross_images_conf.reshape(B, V, 1, H, W)
