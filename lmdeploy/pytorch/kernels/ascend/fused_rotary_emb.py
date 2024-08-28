# Copyright (c) OpenMMLab. All rights reserved.
# import dlinfer.ops as ext_ops
import torch
from torch import Tensor


# def fused_rotary_emb(
#     query_states: Tensor,
#     key_states: Tensor,
#     position_ids: torch.LongTensor,
#     inv_freq: Tensor,
#     scaling_factor: float,
#     out_q: Tensor = None,
#     out_k: Tensor = None,
#     context=None,
# ):
#     batch, seqlen, head, dim = query_states.shape
#     num_kv_heads = key_states.shape[-2]
#     query_states_reshaped = query_states.view(batch, seqlen, head, dim)
#     key_states_reshaped = key_states.view(batch, seqlen, num_kv_heads, dim)
#     position_ids = position_ids.squeeze(0).unsqueeze(-1)
#     pos_freq = position_ids / scaling_factor * inv_freq
#     if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
#         cos = (torch.cos(pos_freq).view(batch, seqlen, 1,
#                                         -1).repeat(1, 1, 1,
#                                                    2).to(query_states.dtype))
#         sin = (torch.sin(pos_freq).view(batch, seqlen, 1,
#                                         -1).repeat(1, 1, 1,
#                                                    2).to(query_states.dtype))
#         if context:
#             setattr(context, 'cos', cos)
#             setattr(context, 'sin', sin)
#     cached_cos = context.cos if context else cos
#     cached_sin = context.sin if context else sin
#     ext_ops.apply_rotary_pos_emb(query_states_reshaped, key_states_reshaped,
#                                  cached_cos, cached_sin, None, None)
#     if out_q is None:
#         out_q = query_states
#     else:
#         out_q.copy_(query_states)
#     if out_k is None:
#         out_k = key_states
#     else:
#         out_k.copy_(key_states)
#     return out_q, out_k


def fused_rotary_emb(q: Tensor,
                     k: Tensor,
                     position_ids: torch.LongTensor,
                     inv_freq: Tensor,
                     scaling_factor: float,
                     out_q: Tensor = None,
                     out_k: Tensor = None,
                     context=None):
    """Fuse `rotary_embedding` and `apply_rotary_pos_emb`."""
    if out_q is None:
        out_q = torch.empty_like(q)
    else:
        assert q.stride() == out_q.stride()
    if out_k is None:
        out_k = torch.empty_like(k)
    else:
        assert k.stride() == out_k.stride()
    _, seq_len, _, _ = q.size()
    for i in range(seq_len):
        position_id = position_ids[:, i] / scaling_factor
        pos_freq = position_id * inv_freq
        cos = torch.cos(pos_freq).to(q.dtype)
        sin = torch.sin(pos_freq).to(q.dtype)
        _, seq_len, _, dim = q.shape
        q0 = q[:, i, :, 0: dim // 2]
        q1 = q[:, i, :, dim // 2: dim]
        out_q0 = q0 * cos - q1 * sin
        out_q1 = q0 * sin + q1 * cos
        out_q[:, i] = torch.cat((out_q0, out_q1), dim=-1)

        _, seq_len, _, dim = k.shape
        k0 = k[:, i, :, 0: dim // 2]
        k1 = k[:, i, :, dim // 2: dim]
        out_k0 = k0 * cos - k1 * sin
        out_k1 = k0 * sin + k1 * cos
        out_k[:, i] = torch.cat((out_k0, out_k1), dim=-1)

    return out_q, out_k
