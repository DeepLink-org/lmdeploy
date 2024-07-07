import torch
from torch import Tensor
from torch import nn
from . import apply_rotary_pos_emb

import vllm._C as vllm_ops

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def fused_rotary_emb(
    query_states: Tensor,
    key_states: Tensor,
    position_ids: torch.LongTensor,
    inv_freq: Tensor,
    scaling_factor: float,
    out_q: Tensor = None,
    out_k: Tensor = None,
    context=None,
):
    bs, seq_len, head, dim = query_states.shape
    _, _, numKeyValueHeads, _ = key_states.shape
    # query_states = query_states.reshape(-1, head * dim)
    # key_states = key_states.reshape(-1, numKeyValueHeads * dim)

    position_ids = position_ids.squeeze(0).unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    cos = torch.cos(pos_freq).view(1, position_ids.shape[0], -1).repeat(1, 1, 2).to(query_states.dtype)
    sin = torch.sin(pos_freq).view(1, position_ids.shape[0], -1).repeat(1, 1, 2).to(query_states.dtype)
    # import pdb; pdb.set_trace()

    cos = cos.to(query_states.device).to(query_states.dtype)
    sin = sin.to(query_states.device).to(query_states.dtype)

    # cos = cos.unsqueeze(2)
    # cos = cos[..., :cos.shape[-1] // 2]
    # sin = sin.squeeze(2)
    # sin = sin[..., :sin.shape[-1] // 2]
    # cos_sin_cache = torch.cat((cos, sin), dim=-1)

    # vllm_ops.ops.rotary_embedding(position_ids,
    #                             query_states,
    #                             key_states,
    #                             dim,
    #                             cos_sin_cache,
    #                             True)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # query_states = query_states.view(bs, seq_len, head, dim)
    # key_states = key_states.view(bs, seq_len, numKeyValueHeads, dim)

    return query_states, key_states
