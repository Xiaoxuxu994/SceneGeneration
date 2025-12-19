import torch
import torch.nn as nn

from .utils import init_weights

class MLP(nn.Module):
    
    """
    Multi-Layer Perceptron block.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    """
    
    def __init__(
        self, 
        d_in, 
        d_out, 
        mlp_ratio=4., 
        bias=False, 
        dropout=0.0, 
        activation=nn.GELU, 
        mlp_dim=None, 
    ):
        """Initialize MLP.

        Args:
            d_in (int): Input dimension. 
            d_out (int): Output dimension. 
            mlp_ratio (float, optional): Ratio to expand input dimension to hidden layer. Defaults to 4.
            bias (bool, optional): If enable bias in linear layer. Defaults to False.
            dropout (float, optional): Dropout ratio. Defaults to 0.0.
            activation (_type_, optional): Activation function. Defaults to nn.GELU.
            mlp_dim (_type_, optional): Hidden layer dimension. Defaults to None.
        """
        super().__init__()
        hidden_dim = mlp_dim if mlp_dim is not None else int(d_in * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim, bias=bias), # .apply(init_weights), 
            activation(), 
            nn.Linear(hidden_dim, d_out, bias=bias), # .apply(init_weights), 
            nn.Dropout(dropout), 
        )
    
    def forward(self, x):
        return self.mlp(x)
