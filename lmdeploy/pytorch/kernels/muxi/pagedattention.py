# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import math
from torch import Tensor
from flash_attn import flash_attn_varlen_func

import vllm._C as vllm_ops

def make_cu_seqlens(seqlens):
    cu_seqlens = seqlens.cumsum(0)
    cu_zero = cu_seqlens.new_zeros(1)
    cu_seqlens = torch.cat([cu_zero, cu_seqlens])
    return cu_seqlens

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

    # import pdb; pdb.set_trace()
    is_decoding = query_states.shape[-3] == q_seqlens.size(0)
    if not is_decoding:
        _, head, dim = key_states.size()
        # if context and hasattr(context, 'cu_seqlens_q'):
        cu_seqlens_q = context.cu_seqlens_q
        cu_seqlens_kv = context.cu_seqlens_kv
        # else:
        #     cu_seqlens_q = make_cu_seqlens(q_seqlens).int()
        #     cu_seqlens_kv = make_cu_seqlens(kv_seqlens).int()
        #     max_seqlen_q = q_seqlens.max().item()
        #     max_seqlen_kv = kv_seqlens.max().item()
 
        if window_size:
            win_size = (window_size, window_size)
        else:
            win_size = (-1, -1)
        attn_output.copy_(flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_q_seq_length,
            max_seqlen_k=max_kv_seq_length,
            softmax_scale=float(1 / math.sqrt(dim)),
            causal=True,
            window_size=win_size,
        ))
    else:
        # if context and hasattr(context, 'max_kv_seq_length'):
        # max_seqlen_kv = context.max_kv_seq_length
        # else:
        #     max_seqlen_kv = kv_seqlens.max().item()
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
