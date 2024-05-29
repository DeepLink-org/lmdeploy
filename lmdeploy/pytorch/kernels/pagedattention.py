# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import math
import deeplink_ext.cpp_extensions as ext
from torch import Tensor


mask_cache = {}


def flash_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
    batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
    numKeyValueHeads = k.shape[1]
    assert k.shape[1] == v.shape[1]
    scale = 1 / math.sqrt(dim)
    for i in range(batch):
        start = b_start_loc[i]
        end = start + b_seq_len[i]

        single_seq_len = int(b_seq_len[i])
        single_q = q[start:end].view(1, single_seq_len, -1)
        single_k = k[start:end].view(1, single_seq_len, -1)
        single_v = v[start:end].view(1, single_seq_len, -1)

        single_out = out[start:end, :].view(1, single_seq_len, -1)
        if single_seq_len not in mask_cache:
            mask = torch.tril(torch.ones(single_seq_len, single_seq_len, dtype=torch.bool), diagonal=0).cuda()
            mask = mask.repeat(1, 1, 1)
            mask = torch.logical_not(mask)
            mask_cache[single_seq_len] = mask
            print(f"cache mask in context attention, seqLen:{single_seq_len}")
        mask = mask_cache[single_seq_len]
        ext.prompt_flash_attention(single_out, single_q, single_k, single_v, None, mask, [], head, scale, 2147473647, 0, "BSH", numKeyValueHeads)
    return out


def fused_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
    batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
    numKeyValueHeads = k.shape[1]
    assert k.shape[1] == v.shape[1]
    scale = 1.0 / math.sqrt(dim)

    mask_key_str = str(batch) + ":" + str(max_input_len)
    if mask_key_str not in mask_cache:
        mask = torch.tril(torch.ones(max_input_len, max_input_len, dtype=torch.bool), diagonal=0).cuda()
        mask = mask.repeat(batch, 1, 1)
        mask = torch.logical_not(mask)
        mask_cache[mask_key_str] = mask
        print(f"cache mask in context attention, batch:seqLen={mask_key_str}")
    
    mask = mask_cache[mask_key_str]
    ext.prompt_flash_attention(
        out.view(batch, max_input_len, head*dim), 
        q.view(batch, max_input_len, head*dim), 
        k.view(batch, max_input_len, numKeyValueHeads*dim), 
        v.view(batch, max_input_len, numKeyValueHeads*dim), 
        None, mask, b_seq_len, head, scale, 2147473647, 0, "BSH", numKeyValueHeads)
    return out


def paged_token_attention(q, k_cache, v_cache, out, b_seq_len, block_table:torch.Tensor, block_size):
    numKeyValueHeads = k_cache.shape[1]
    assert k_cache.shape[1] == v_cache.shape[1]
    batch, head, dim = q.shape
    kv_cache_len = k_cache.shape[0]
    q = q.reshape(batch, head*dim).unsqueeze(1)
    k_cache = k_cache.reshape(kv_cache_len, numKeyValueHeads*dim).unsqueeze(0)
    v_cache = v_cache.reshape(kv_cache_len, numKeyValueHeads*dim).unsqueeze(0)
    out = out.view(q.shape)
    ext.paged_attention(out, q, k_cache, v_cache,
                        None, None, 
                        b_seq_len, block_table, head, numKeyValueHeads,
                        1.0 / math.sqrt(dim), "BSH", block_size, 0, 
                        None, None, None, None, None, None, None, None
                        )
    return out


def paged_attention_fwd(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = None,
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
        return flash_context_attention(query_states, key_states, value_states, attn_output,
                                       q_start_loc, q_seqlens.tolist(), max_seqlen)
    else:
        block_num, block_size, head, dim = key_cache.size()
        k = key_cache.reshape(block_num * block_size, head, dim)
        v = value_cache.reshape(block_num * block_size, head, dim)
        return paged_token_attention(query_states, k, v, attn_output, kv_seqlens.tolist(),
                                     block_offsets.to(torch.int32), block_size)
