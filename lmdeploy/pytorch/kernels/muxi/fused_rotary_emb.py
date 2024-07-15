import torch
from torch import nn
from torch import Tensor

from vllm._C import ops


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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
    _, seq_len, head, dim = query_states.shape
    kv_head = key_states.shape[2]
    query_states = query_states.reshape(seq_len, head * dim)
    key_states = key_states.reshape(seq_len, kv_head * dim)

    position_ids = position_ids.squeeze(0).unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    cos = torch.cos(pos_freq).view(seq_len, -1).repeat(1, 2).to(query_states.dtype)
    sin = torch.sin(pos_freq).view(seq_len, -1).repeat(1, 2).to(query_states.dtype)

    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    new_cos = cos.squeeze(-2)
    new_cos = new_cos[..., :new_cos.shape[-1] // 2]
    new_sin = sin.squeeze(-2)
    new_sin = new_sin[..., :new_sin.shape[-1] // 2]
    cos_sin_cache = torch.cat((new_cos, new_sin), dim=-1)

    ops.rotary_embedding(position_ids.view(seq_len),
                        query_states,
                        key_states,
                        dim,
                        cos_sin_cache,
                        True)
    return query_states.view(seq_len, head, dim), key_states.view(seq_len, kv_head, dim)
