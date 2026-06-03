import torch.nn as nn
import torch.nn.functional as F


def linear_in_dtype(x, linear):
    """Run an ``nn.Linear`` in the dtype of its input ``x``.

    Casts the layer's weight/bias to ``x.dtype`` (instead of casting the input to the
    weight dtype). This lets the *call site* decide the precision: pass an fp32 input
    and the layer runs in fp32 even when its stored weights are bf16 (e.g. under
    DeepSpeed native bf16); pass a bf16 input and it runs in bf16.

    Gradients flow back to the original (bf16) parameters via the cast; the module is
    not mutated and no full-module copy is made.

    Note: under ZeRO-3 a parameter is sharded outside its gather context, so reading
    ``linear.weight`` directly (here or anywhere) needs a gather context. This is fine
    under ZeRO-1/2 where parameters are resident.
    """
    assert isinstance(linear, nn.Linear), (
        f"linear_in_dtype expects an nn.Linear, got {type(linear).__name__}"
    )
    bias = None if linear.bias is None else linear.bias.to(x.dtype)
    return F.linear(x, linear.weight.to(x.dtype), bias)