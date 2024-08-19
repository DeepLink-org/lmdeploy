# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor
from typing import Optional
import infer_ext.ops as ext_ops

def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    residual: Optional[Tensor] = None,
):
    if residual is None:
        return ext_ops.rms_norm(hidden_states, weight, eps)
    else:
        return ext_ops.add_rms_norm(hidden_states, residual, weight, eps)
