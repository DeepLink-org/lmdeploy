# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q_states: Tensor, k_states: Tensor, cached_cos: Tensor, cached_sin: Tensor, position_ids: Tensor, position_ids_1d: Tensor, q_embed=None, k_embed=None, context=None):
    cos = cached_cos[position_ids_1d, None, :]
    sin = cached_sin[position_ids_1d, None, :]

    q_embed = q_states * cos + _rotate_half(q_states) * sin
    k_embed = k_states * cos + _rotate_half(k_states) * sin

    return q_embed, k_embed
