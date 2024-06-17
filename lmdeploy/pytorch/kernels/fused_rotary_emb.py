import torch
from torch import Tensor
import deeplink_ext.cpp_extensions as ext


def fused_rotary_emb(q: Tensor,
                     k: Tensor,
                     position_ids: torch.LongTensor,
                     inv_freq: Tensor,
                     scaling_factor: float,
                     out_q: Tensor = None,
                     out_k: Tensor = None):
    seq_len, head, dim = q.shape
    q = q.view(seq_len, head*dim)
    k = k.view(seq_len, head*dim)
    position_ids = position_ids.unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    cos = torch.cos(pos_freq).view(position_ids.shape[0], 1, -1).repeat(1,1,2).to(q.dtype)
    sin = torch.sin(pos_freq).view(position_ids.shape[0], 1, -1).repeat(1,1,2).to(q.dtype)
    ext.rotary_embedding_v2(q, k, cos, sin, dim)
    return q.view(seq_len, head, dim), k.view(seq_len, head, dim)
