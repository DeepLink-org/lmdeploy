import torch
from torch import Tensor
import deeplink_ext.cpp_extensions as ext


def rotary_emb(q, cos, sin, out_q):
    out_q = q if out_q is None else out_q
    seq_len, _, dim = q.shape
    cos_view = cos.view([seq_len, 1, dim // 2])
    sin_view = sin.view([seq_len, 1, dim // 2])
    ext.apply_rotary(out_q, q, cos_view, sin_view, False, False)


def fused_rotary_emb(q: Tensor,
                     k: Tensor,
                     position_ids: torch.LongTensor,
                     inv_freq: Tensor,
                     scaling_factor: float,
                     out_q: Tensor = None,
                     out_k: Tensor = None):
    position_ids = position_ids.unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    cos = torch.cos(pos_freq).to(q.dtype)
    sin = torch.sin(pos_freq).to(q.dtype)
    rotary_emb(q, cos, sin, out_q)
    rotary_emb(k, cos, sin, out_k)
    return out_q, out_k
