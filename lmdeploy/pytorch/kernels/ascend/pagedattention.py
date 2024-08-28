# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
import torch
import math
from torch import Tensor


# def prefill_attention(
#     query_states: Tensor,
#     key_states: Tensor,
#     value_states: Tensor,
#     attn_output: Tensor,
#     key_cache: Tensor,
#     value_cache: Tensor,
#     block_offsets: Tensor,
#     q_start_loc: Tensor,
#     q_seq_len: Tensor,
#     kv_seq_len: Tensor,
#     block_size: int,
#     kv_cache_len: int,
#     context=None,
# ):
#     num_q_heads, dim = query_states.shape[1:3]
#     num_kv_heads = value_states.shape[1]

#     if context.is_unpaged_prefill:
#         ext_ops.prefill_attention(
#             query_states,
#             key_states,
#             value_states,
#             q_start_loc,
#             q_seq_len,
#             context.max_q_seq_length,
#             num_q_heads,
#             num_kv_heads,
#             attn_mask=context.attention_mask,
#             attn_output=attn_output,
#         )
#     else:
#         key_cache = key_cache.reshape(1, kv_cache_len, num_kv_heads * dim)
#         value_cache = value_cache.reshape(1, kv_cache_len, num_kv_heads * dim)
#         ext_ops.paged_prefill_attention(
#             query_states,
#             key_cache,
#             value_cache,
#             block_offsets,
#             block_size,
#             q_start_loc,
#             q_seq_len,
#             kv_seq_len,
#             num_q_heads,
#             num_kv_heads,
#             attn_mask=context.attention_mask,
#             attn_output=attn_output,
#         )


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
    block_size: int,
    kv_cache_len: int,
    context=None,
):
    batch, head, dim = q_start_loc.shape[0], query_states.shape[1], query_states.shape[2]
    numKeyValueHeads = key_states.shape[1]
    assert key_states.shape[1] == value_states.shape[1]
    # scale = 1 / math.sqrt(dim)
    for i in range(batch):
        start = q_start_loc[i]
        end = start + q_seq_len[i]

        single_seq_len = int(q_seq_len[i])
        single_q = query_states[start:end].view(single_seq_len, head, dim).transpose(0, 1)
        single_k = key_states[start:end].view(single_seq_len, numKeyValueHeads, dim).transpose(0, 1)
        single_v = value_states[start:end].view(single_seq_len, numKeyValueHeads, dim).transpose(0, 1)

        single_out = attn_output[start:end, :].view(single_seq_len, head, dim)
        # if single_seq_len not in mask_cache:
        #     mask = torch.tril(torch.ones(single_seq_len, single_seq_len, dtype=torch.bool), diagonal=0).cuda()
        #     mask = mask.repeat(1, 1, 1)
        #     mask = torch.logical_not(mask)
        #     mask_cache[single_seq_len] = mask
        #     print(f"cache mask in context attention, seqLen:{single_seq_len}")
        mask = context.attention_mask[i]

        attn_weights = torch.matmul(single_q, single_k.transpose(-2, -1)) / math.sqrt(dim)
        if mask is not None:
            attn_weights += mask.unsqueeze(0)

        attn_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        single_out[:] = torch.matmul(attn_probs, single_v).transpose(0, 1).contiguous()
        # ext.prompt_flash_attention(single_out, single_q, single_k, single_v, None, mask, [], head, scale, 2147473647, 0, "BSH", numKeyValueHeads)
    # return out
    
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    # if context.attention_mask is not None:  # no matter the length, we just slice it
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask

    # # upcast attention to fp32
    # attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # attn_output[:] = torch.matmul(attn_weights, value_states)


def paged_decode_attention(q, k_cache, v_cache, attn_output, kv_seq_len,
                           max_kv_seq_len, block_offsets, block_size):
    num_kv_heads, num_q_heads = k_cache.shape[1], q.shape[1]
    ext_ops.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_output=attn_output.view(q.shape),
    )


def paged_attention_fwd(
    query_states: Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = 1,
    context=None,
):
    is_decoding = query_states.shape[-3] == q_seqlens.size(0)
    block_num, block_size, head, dim = key_cache.size()
    kv_cache_len = block_num * block_size
    k = key_cache.reshape(block_num * block_size, head, dim)
    v = value_cache.reshape(block_num * block_size, head, dim)
    if not is_decoding:
        prefill_attention(
            query_states,
            key_states,
            value_states,
            attn_output,
            k,
            v,
            block_offsets,
            q_start_loc,
            q_seqlens,
            kv_seqlens,
            block_size,
            kv_cache_len,
            context=context,
        )
    else:
        paged_decode_attention(
            query_states,
            k,
            v,
            attn_output,
            kv_seqlens,
            context.max_kv_seq_length,
            block_offsets,
            block_size,
        )
