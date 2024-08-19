# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional
from torch import Tensor

import infer_ext.ops as ext_ops

def paged_attention_fwd(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_q_seq_length: int,
    max_kv_seq_length: int,
    window_size: Optional[int] = None,
    context=None,
):
    """Paged Attention forward.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        q_start_loc (Tensor): Start token location of each data in batch.
        q_seqlens (Tensor): Query length for each data in batch.
        kv_seqlens (Tensor): Key/Value length for each data in batch.
        max_seqlen (int): The max input length.
        BLOCK (int): The kernel block size.
    """

    if not context.is_decoding and query_states.size(0) == key_states.size(0):
        _, num_kv_heads, dim = key_states.size()
        if not hasattr(context, "softmax_scale"):
            setattr(context, "softmax_scale", float(1 / math.sqrt(dim)))
        if window_size is None:
            window_size = -1
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        query_states, key_states, value_states = \
            [maybe_contiguous(x) for x in (query_states, key_states, value_states)]

        ext_ops.context_attention(
            query_states,
            key_states,
            value_states,
            q_start_loc,
            q_seqlens,
            max_q_seq_length,
            0, # num_q_heads, not used
            0, # num_kv_heads, not used
            None, # attn_mask, not used, force causal mask internal
            softmax_scale=context.softmax_scale,
            attn_output=attn_output,
        )
    else:
        _, num_kv_heads, block_size, dim = value_cache.size()
        if not hasattr(context, "softmax_scale"):
            setattr(context, "softmax_scale", float(1 / math.sqrt(dim)))
        return ext_ops.paged_decode_attention(
            query_states,
            key_cache,
            value_cache,
            block_offsets,
            block_size,
            kv_seqlens,
            max_kv_seq_length,
            0, # num_q_heads, not used
            num_kv_heads,
            softmax_scale=context.softmax_scale,
            attn_output=attn_output,
        )
