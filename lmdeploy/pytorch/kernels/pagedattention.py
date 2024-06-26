# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import math
#import deeplink_ext.cpp_extensions as ext
from torch import Tensor
from flash_attn import flash_attn_varlen_func

import vllm._C as vllm_ops


mask_cache = {}


def flash_context_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: list,
    kv_seqlens: list,
    block_size: int,
    kv_cache_len: int,
):
    batch, head, dim = q_start_loc.shape[0], query_states.shape[1], query_states.shape[2]
    numKeyValueHeads = value_states.shape[1]
    assert key_states.shape[1] == value_states.shape[1]
    for i in range(batch):
        start = q_start_loc[i]
        end = start + q_seqlens[i]
        single_seqlen = int(end - start)
        single_q = query_states[start:end].view(1, single_seqlen, -1)
        single_k = key_states[start:end].reshape(1, single_seqlen, -1)
        single_v = value_states[start:end].reshape(1, single_seqlen, -1)
        single_out = attn_output[start:end, :].view(1, single_seqlen, -1)
        if q_seqlens[i] == kv_seqlens[i]:
            if single_seqlen not in mask_cache:
                mask = torch.tril(torch.ones(single_seqlen, single_seqlen, dtype=torch.bool), diagonal=0).cuda()
                mask = torch.logical_not(mask)
                mask_cache[single_seqlen] = mask
                print(f"cache mask in context attention, seqLen:{single_seqlen}")
            mask = mask_cache[single_seqlen]
            ext.prompt_flash_attention(single_out, single_q, single_k, single_v,
                                       mask, [kv_seqlens[i]], kv_seqlens[i], head, numKeyValueHeads, dim)
        else:
            key_cache = key_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
            value_cache = value_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
            for j in range(q_seqlens[i]):
                single_q = query_states[start+j:start+j+1].view(1, 1, -1)
                single_out = attn_output[start+j:start+j+1].view(1, 1, -1)
                if f"{q_seqlens[i]}_{kv_seqlens[i]}" not in mask_cache:
                    mask = torch.tril(torch.ones(q_seqlens[i], kv_seqlens[i], dtype=torch.bool), diagonal=kv_seqlens[i] - q_seqlens[i]).cuda()
                    mask = torch.logical_not(mask)
                    mask_cache[f"{q_seqlens[i]}_{kv_seqlens[i]}"] = mask
                    print(f"cache mask in context attention, seqLen:{q_seqlens[i]}_{kv_seqlens[i]}")
                mask = mask_cache[f"{q_seqlens[i]}_{kv_seqlens[i]}"]
                ext.paged_attention(single_out, single_q, key_cache, value_cache, mask[j:j+1],
                                    [kv_seqlens[i]],  head, numKeyValueHeads,
                                    dim, block_offsets[i:i+1], block_size
                                    )


def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seqlens, block_table:torch.Tensor, block_size):
    numKeyValueHeads = k_cache.shape[1]
    assert k_cache.shape[1] == v_cache.shape[1]
    bs, head, dim = q.shape
    kv_cache_len = k_cache.shape[0]
    q = q.reshape(bs, 1, head*dim)
    k_cache = k_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
    v_cache = v_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
    ext.paged_attention(attn_output.view(q.shape), q, k_cache, v_cache, None,
                        kv_seqlens, head, numKeyValueHeads,
                        dim, block_table, block_size
                        )

def _make_bias(seq_lens, history_lens, neg_val=-1e30):
    full_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()
    max_full_len = full_seq_lens.max().item()
    seq_ranges = [torch.arange(max_seq_len) for _ in seq_lens]
    for r, l in zip(seq_ranges, seq_lens):
        r[l:] = -max_full_len
    seq_ranges = torch.stack(seq_ranges, dim=0).cuda()
    kv_ranges = [torch.arange(max_full_len) for _ in full_seq_lens]
    kv_ranges = torch.stack(kv_ranges, 0).cuda()
    mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:,
                                                                         None,
                                                                         None]
    return mask.float() * neg_val

def _naive_attention(batched_q, batched_k, batched_v, bias):
    #batched_k, batched_v = batched_kv

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k


    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    # expand group
    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    attn_weight = qk + bias[:, None]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()


    return attn_output.squeeze(1)



def paged_attention_fwd(
    query_states: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = None,
    key_states: Tensor = None,
    value_states: Tensor = None,
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
        block_num, block_size, head, dim = key_cache.size()
        if key_states is None or value_states is None:
            block_num, block_size, head, dim = key_cache.size()
            batch_size = q_seqlens.size(0)
            batch_k = []
            batch_v = []
            for i in range(batch_size):
                offsets = block_offsets[i]
                tmp_k = key_cache[offsets].view(-1, head, dim)
                tmp_k = tmp_k[:kv_seqlens[i]]
                batch_k.append(tmp_k)
      
                tmp_v = value_cache[offsets].view(-1, head, dim)   
                tmp_v = tmp_v[:kv_seqlens[i]]
                batch_v.append(tmp_v)
            key_states = torch.cat(batch_k, 0)
            value_states = torch.cat(batch_v, 0)
        def _make_cu_seqlens(seqlens):
            cu_seqlens = seqlens.cumsum(0)
            cu_zero = cu_seqlens.new_zeros(1)
            cu_seqlens = torch.cat([cu_zero, cu_seqlens])
            return cu_seqlens
        max_seqlen_q = q_seqlens.max().item()
        max_seqlen_k = kv_seqlens.max().item()
        cu_seqlens_q = _make_cu_seqlens(q_seqlens).int()
        cu_seqlens_k = _make_cu_seqlens(kv_seqlens).int()
        if window_size:
            win_size = (window_size, window_size)
        else:
            win_size = (-1, -1)
        attn_output.copy_(flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=float(1 / math.sqrt(dim)),
            causal=True,
            window_size=win_size,
        ))
    else:
        block_num, block_size, head, dim = key_cache.size()
        max_context_len = kv_seqlens.max().item()
        kv_cache_len = block_num * block_size
        x = 32 // key_cache.element_size()
        vllm_key_cache = key_cache.view(block_num, block_size, head, dim // x, x).permute(0, 2, 3, 1, 4).contiguous()  # block_num head block_size dim
        vllm_value_cache = value_cache.transpose(2, 1).contiguous()

        vllm_ops.ops.paged_attention_v1(
            attn_output,
            query_states,
            vllm_key_cache,
            vllm_value_cache,
            head,
            float(1 / math.sqrt(dim)), # scale
            block_offsets.to(torch.int32),
            kv_seqlens.to(torch.int32),
            block_size,
            max_context_len, #max_seqlen,
            None,
            'auto',
        )
