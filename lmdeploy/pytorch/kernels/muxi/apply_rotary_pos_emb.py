# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

import dlinfer.ops as ext_ops

def apply_rotary_pos_emb(q_states: Tensor,
                         k_states: Tensor,
                         cos: Tensor,
                         sin: Tensor,
                         position_ids_1d: Tensor,
                         cos_sin_cache: Tensor):
    ext_ops.apply_rotary_pos_emb(q_states,
                                 k_states,
                                 cos,
                                 sin,
                                 position_ids_1d,
                                 cos_sin_cache)
    return q_states, k_states
