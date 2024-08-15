# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
from torch import Tensor
import torch

def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             epsilon: float = 1e-6,
             residual: Tensor = None,
             out: Tensor = None,
             out_residual: Tensor = None):
    if residual is None:
        return ext_ops.rms_norm(hidden_states, weight, epsilon)
    else:
        add = torch.ops.atb.add(hidden_states, residual)
        norm = ext_ops.rms_norm(add, weight, epsilon)
        return (norm, add)
        # return torch.ops.atb.add_rms_norm(hidden_states, residual, weight, epsilon)
        # # return ext_ops.add_rms_norm(hidden_states, residual, weight, epsilon)
