# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
import math
import torch
from dlinfer.utils.type_annotation import Optional, Sequence, Tensor

def prefill_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    max_q_seq_len: int,
    block_size: int,
    cu_seqlens: Tensor,
    attn_mask: Sequence[Optional[Tensor]],
    is_unpaged_prefill: Optional[bool],
):
    num_q_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    
    if is_unpaged_prefill:
        output = torch.empty_like(query_states)
        ext_ops.prefill_attention(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            q_seq_len,
            max_q_seq_len,
            num_q_heads,
            num_kv_heads,
            attn_mask,
            # softmax_scale=1.0,
            softmax_scale = 1. / math.sqrt(query_states.shape[-1]),
            attn_output=output)
        attn_output.copy_(output)
        return attn_output
    else:
        pass

def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seq_len,
                          max_kv_seq_len, block_offsets, block_size):
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    q = q.unsqueeze(1)
    
    ret = ext_ops.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale = 1. / math.sqrt(q.shape[-1]),
        attn_output = q,
    )
    return q


def paged_attention_fwd(
    query_states: Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    is_decoding: bool,
    block_size: int,
    cu_seqlens: Tensor,
    attn_mask: Sequence[Optional[Tensor]] = (),
    is_unpaged_prefill: Optional[bool] = None,
):
    if not is_decoding:
        return prefill_attention(
            query_states,
            key_states,
            value_states,
            attn_output,
            key_cache,
            value_cache,
            block_offsets,
            q_start_loc,
            q_seqlens,
            kv_seqlens,
            max_q_seq_len,
            block_size,
            cu_seqlens,
            attn_mask,
            is_unpaged_prefill,
        )
       
    else:
        return paged_token_attention(
            query_states,
            key_cache,
            value_cache,
            attn_output,
            kv_seqlens,
            max_kv_seq_len,
            block_offsets,
            block_size,
        )
      
