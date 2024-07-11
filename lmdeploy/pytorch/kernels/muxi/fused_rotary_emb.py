import torch
from torch import Tensor
from torch import nn
from . import apply_rotary_pos_emb

import vllm._C as vllm_ops


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
    position_ids = position_ids.squeeze(0).unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    cos = torch.cos(pos_freq).view(1, position_ids.shape[0], -1).repeat(1, 1, 2).to(query_states.dtype)
    sin = torch.sin(pos_freq).view(1, position_ids.shape[0], -1).repeat(1, 1, 2).to(query_states.dtype)

    cos = cos.to(query_states.device).to(query_states.dtype)
    sin = sin.to(query_states.device).to(query_states.dtype)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    return query_states, key_states
