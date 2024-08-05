# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from vllm._C import ops

def apply_rotary_pos_emb(q_states: Tensor,
                         k_states: Tensor,
                         position_ids_1d: Tensor,
                         dim,
                         context=None):
    ops.rotary_embedding(position_ids_1d,
                        q_states,
                        k_states,
                        dim,
                        context.cos_sin_cache,
                        True)
    return q_states, k_states
