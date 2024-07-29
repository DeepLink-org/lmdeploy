# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
from torch import Tensor


def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             epsilon: float = 1e-6,
             residual: Tensor = None,
             out: Tensor = None,
             out_residual: Tensor = None):
    if residual is None:
        return ext_ops.rms_norm(hidden_states, weight, epsilon)
    else:
        return ext_ops.add_rms_norm(hidden_states, residual, weight, epsilon)
