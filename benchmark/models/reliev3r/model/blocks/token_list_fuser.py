import torch
import torch.nn as nn

from typing import List
from .mlp import MLP



class TokenListFuser(nn.Module):
    
    def __init__(self, d_model, layers=6):
        super().__init__()
        self.d_model = d_model
        self.layers  = layers
        self.blocks  = nn.ModuleList([MLP(d_model, d_model) for _ in range(layers)])
        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
    
    def forward(self, token_list: List[torch.Tensor]):
        """Forward pass of token list fuser. 

        Args:
            token_list (List[torch.Tensor]): List of tensors.
        
        Returns:
            Fused token list (List[torch.Tensor]).
        """
        assert self.layers == len(token_list)
        x = token_list[0]
        for i in range(self.layers):
            x = self.blocks[i](self.norms[i](x))
            if i != self.layers - 1:
                x = x + token_list[i + 1]
        return x
