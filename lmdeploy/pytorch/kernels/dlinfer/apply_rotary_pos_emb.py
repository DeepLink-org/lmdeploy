# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import dlinfer.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Optional[Tensor],
    k_embed: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    query_states, key_states = ext_ops.apply_rotary_pos_emb(query_states, key_states, cos, sin, None, None)

    if q_embed is None:
        q_embed = query_states
    elif q_embed is not query_states:
        q_embed.copy_(query_states)

    if k_embed is None:
        k_embed = key_states
    elif k_embed is not key_states:
        k_embed.copy_(key_states)

    return q_embed, k_embed
