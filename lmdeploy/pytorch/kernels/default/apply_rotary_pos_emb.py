# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import LongTensor, Tensor

def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
    position_ids_1d: Tensor,
    q_embed=None,
    k_embed=None,
    context=None,
):
    bs, head, dim = query_states.shape
    numKeyValueHeads = key_states.shape[1]
    batch, max_seqlen = position_ids.shape
    seqlens =[(min(position_id), max(position_id)+ 1) for position_id in position_ids.tolist()]
    #q=q.reshape(bs, head*dim)
    #k=k.reshape(bs, numKeyValueHeads*dim)
    cos =torch.cat([cos[i:j] for i, j in seqlens])
    sin =torch.cat([sin[i:j] for i, j in seqlens])
    # ext.rotary embedding v2(q, k, cos, sin, dim)
    for i in range(bs):
        q0 = query_states[i, :, :dim//2]
        q1 = query_states[i, :, dim//2:]
        outq0 = q0 * cos[i, :dim//2] - q1 * sin[i, :dim//2]
        outq1 = q0 * sin[i, :dim//2] + q1 * cos[i, :dim//2]
        query_states[i]= torch.cat((outq0, outq1), dim=-1)
        k0 = key_states[i:i+1, :, :dim//2]
        k1 = key_states[i:i+1, :, dim//2:]
        outk0 = k0 * cos[i, :dim//2]- k1 * sin[i, :dim//2]
        outk1 = k0 * sin[i, :dim//2]+ k1 * cos[i, :dim//2]
        key_states[i] = torch.cat((outk0, outk1),dim=-1)
    return query_states.view(bs, head, dim), key_states.view(bs, numKeyValueHeads, dim)