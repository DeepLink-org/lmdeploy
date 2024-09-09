# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
from torch import Tensor

def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float = 1e-6, out: Tensor = None):
    # hidden_states [total_seq_len, hidden_size]
    # weight [hidden_size]
    output =  ext_ops.rms_norm(hidden_states, weight, epsilon)
    
    if out is None:
        out = output
    else:
        out.copy_(output)
    return out
