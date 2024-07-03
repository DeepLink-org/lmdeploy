# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from vllm._C import ops

def apply_rotary_pos_emb(q_states: Tensor,
                         k_states: Tensor,
                         cached_cos: Tensor,
                         cached_sin: Tensor,
                         position_ids: Tensor,
                         position_ids_1d: Tensor,
                         q_embed=None,
                         k_embed=None,
                         context=None):
    bs, head, dim = q_states.shape
    kv_head = k_states.shape[1]
    q_states = q_states.reshape(bs, head * dim)
    k_states = k_states.reshape(bs, kv_head * dim)

    new_cos = cached_cos.squeeze(-2)
    new_cos = new_cos[..., :new_cos.shape[-1] // 2]
    new_sin = cached_sin.squeeze(-2)
    new_sin = new_sin[..., :new_sin.shape[-1] // 2]
    cos_sin_cache = torch.cat((new_cos, new_sin), dim=-1)

    ops.rotary_embedding(position_ids_1d,
                        q_states,
                        k_states,
                        dim,
                        cos_sin_cache,
                        True)
    return q_states.view(bs, head, dim), k_states.view(bs, kv_head, dim)
