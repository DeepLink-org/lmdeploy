# Copyright (c) OpenMMLab. All rights reserved.
import torch
# import dlinfer.ops as ext_ops
from torch import Tensor


# def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float = 1e-6):
#     return ext_ops.rms_norm(hidden_states, weight, epsilon)


from torch import Tensor
def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    """rms norm."""
    hidden_states_f = hidden_states.to(torch.float32)
    return (hidden_states_f * torch.rsqrt(hidden_states_f.pow(2).mean(-1, keepdim=True) + eps)).to(hidden_states.dtype) * weight
