import torch
from torch import nn
from torch import Tensor

from vllm._C import ops
from torch.profiler import profile, record_function, ProfilerActivity

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # import pdb; pdb.set_trace()
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def fused_rotary_emb(
    query_states: Tensor,
    key_states: Tensor,
    position_ids: torch.LongTensor,
    head_dim: int,
    context=None,
):
    # import pdb; pdb.set_trace()
    ops.rotary_embedding(position_ids,
                        query_states,
                        key_states,
                        head_dim,
                        context.cos_sin_cache,
                        True)
    # import pdb; pdb.set_trace()

    return query_states, key_states

def fused_rotary_emb_eager(
    query_states: Tensor,
    key_states: Tensor,
    position_ids: torch.LongTensor,
    head_dim: int,
    context=None,
):
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, context.cos, context.sin, position_ids)
    return query_states, key_states