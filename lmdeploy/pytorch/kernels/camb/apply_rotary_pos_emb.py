# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed=None,
    k_embed=None,
    cos_sin_ids=None,
    cu_seqlens=None,
):
    query_states, key_states = ext_ops.apply_rotary_pos_emb(query_states, key_states, cos, sin, None, cos_sin_ids, cu_seqlens)
    if q_embed is None or q_embed.data_ptr() == query_states.data_ptr():
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None or k_embed.data_ptr() == key_states.data_ptr():
        k_embed = key_states
    else:
        k_embed.copy_(key_states)
    return q_embed, k_embed
