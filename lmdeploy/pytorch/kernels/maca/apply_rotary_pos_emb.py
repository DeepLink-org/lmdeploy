# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor
import infer_ext.ops as ext_ops

def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    position_ids_1d: Tensor,
    context=None
):
    return ext_ops.apply_rotary_pos_emb(query_states, key_states, None, None,
                                        position_ids_1d, context.cos_sin_cache)
