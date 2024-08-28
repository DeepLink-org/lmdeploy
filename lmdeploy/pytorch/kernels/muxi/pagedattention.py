# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import math
from torch import Tensor

import vllm._C as vllm_ops
from maca_extension import ops as ext_ops
from flash_attn import flash_attn_varlen_func

def make_cu_seqlens(seqlens):
    cu_seqlens = seqlens.cumsum(0)
    cu_zero = cu_seqlens.new_zeros(1)
    cu_seqlens = torch.cat([cu_zero, cu_seqlens])
    return cu_seqlens

# TODO: support maca_extension flash_attn_varlen_func
def ext_flash_attn_varlen_func(
    query_states,
    key_states,
    value_states,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    return_softmax=False,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    query_states, key_states, value_states = [maybe_contiguous(x) for x in (query_states, key_states, value_states)]

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = ext_ops.flash_attn_varlen_fwd(
        query_states,
        key_states,
        value_states,
        None,
        cu_seqlens_q,
        cu_seqlens_kv,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        return_softmax,
        None,
    )

    return out

def paged_attention_fwd_prefill(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_q_seq_length: int,
    max_kv_seq_length: int,
    window_size: None,
    context: None,
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
    # TODO: support maca_extension paged_attention_fwd_prefill
    # return

    _, head, dim = key_states.size()
    # if context and hasattr(context, 'cu_seqlens_q'):
    cu_seqlens_q = context.cu_seqlens_q
    cu_seqlens_kv = context.cu_seqlens_kv
    # else:
    #     cu_seqlens_q = make_cu_seqlens(q_seqlens).int()
    #     cu_seqlens_kv = make_cu_seqlens(kv_seqlens).int()

    if window_size:
        win_size = (window_size, window_size)
    else:
        win_size = (-1, -1)
    attn_output = ext_flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_q_seq_length,
        max_kv_seq_length,
        softmax_scale=float(1 / math.sqrt(dim)),
        causal=True,
        window_size=win_size,
    )

    return attn_output

def paged_attention_fwd(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_q_seq_length: int,
    max_kv_seq_length: int,
    window_size: None,
    context: None,
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
    is_decoding = query_states.shape[-3] == q_seqlens.size(0)
    if not is_decoding:
        _, head, dim = key_states.size()
        # if context and hasattr(context, 'cu_seqlens_q'):
        cu_seqlens_q = context.cu_seqlens_q
        cu_seqlens_kv = context.cu_seqlens_kv
        # else:
        #     cu_seqlens_q = make_cu_seqlens(q_seqlens).int()
        #     cu_seqlens_kv = make_cu_seqlens(kv_seqlens).int()
 
        if window_size:
            win_size = (window_size, window_size)
        else:
            win_size = (-1, -1)
        attn_output.copy_(flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q,
            cu_seqlens_q,
            max_q_seq_length,
            max_q_seq_length,
            softmax_scale=float(1 / math.sqrt(dim)),
            causal=True,
            window_size=win_size,
        ))
    else:
        block_num, head, block_size, dim = value_cache.size()
        vllm_ops.ops.paged_attention_v1(
            attn_output,
            query_states,
            key_cache,
            value_cache,
            head,
            float(1 / math.sqrt(dim)), # scale
            block_offsets,
            kv_seqlens,
            block_size,
            max_kv_seq_length,
            None,
            'auto',
        )
