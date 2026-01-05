import torch
import torch.nn as nn

from einops import rearrange

from .utils import init_weights



class LinearTokenizer(nn.Module):
    
    def __init__(self, d_in, d_model, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(d_in * patch_size**2, d_model, bias=False) # .apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize the input multi-view images into tokens.
        Args:
            x (torch.Tensor): Input multi-view images, tensor of shape (b v c h w).
        Returns:
            torch.Tensor: Embedded tokens, tensor of shape ((b v) l c).
        """
        x = rearrange(x, "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)", ph=self.patch_size, pw=self.patch_size)
        x = self.embedding(x)
        return x


class SinTokenizer(nn.Module):
    
    def __init__(self, d_in, d_model, patch_size):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This tokenizer is designed to analytically 

        Args:
            x (torch.Tensor): Plucker map, tensor of shape (B, V, 6, H, W).

        Returns:
            torch.Tensor: _description_
        """
        
        pass


class LinearUntokenizer(nn.Module):
    """Linear untokenizer for transformer output.
    """
    
    def __init__(self, patch_size, d_feat, d_out, activation="exp", pred_conf=True):
        super().__init__()
        self.patch_size = patch_size
        self.pred_conf  = pred_conf
        if self.pred_conf:
            d_out += 1
        self.d_out = d_out
        self.norm = nn.LayerNorm(d_feat, bias=False)
        self.proj = nn.Linear(d_feat, d_out * self.patch_size**2)
        if activation == "exp":
            self.activate = torch.exp
        elif activation == "identity":
            self.activate = nn.Identity()
        elif activation == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            raise NotImplementedError(f"Activation type {activation} is not implemented.")
    
    def forward(self, tokens: torch.Tensor, img_shape) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): Tokens to depatchify, tensor of shape (B, V, L, D)
        """
        H, W = img_shape
        B, V, S, D = tokens.shape
        
        feat = self.proj(self.norm(tokens)) # B, V, L, d_out * patch_size**2
        feat = self.activate(feat)
        feat = feat.transpose(-1, -2).view(B, V, -1, H // self.patch_size, W // self.patch_size)
        
        """Use pixel shuffle to upsample token to pixels.
        """
        feat = torch.nn.functional.pixel_shuffle(feat, self.patch_size) # B, V, d_out, H, W
        feat = feat.permute(0, 1, 3, 4, 2)
        
        if self.pred_conf:
            return feat[..., :-1], feat[..., -1:]
        
        return feat
        