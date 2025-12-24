import torch
import torch.nn as nn
from . import SparseTensor
import torch.nn.functional as F

__all__ = [
    'SparseLinear',
    'SparseDropout'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))

class SparseDropout(nn.Dropout):
    def __init__(self, p: float = 0.1):
        """
        SparseDropout applies dropout only to the `.feats` field of a SparseTensor.
        """
        super(SparseDropout, self).__init__(p)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
