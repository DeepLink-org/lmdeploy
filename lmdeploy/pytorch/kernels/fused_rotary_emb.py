import torch
from torch import Tensor


def fused_rotary_emb(q: Tensor,
                     k: Tensor,
                     position_ids: torch.LongTensor,
                     inv_freq: Tensor,
                     scaling_factor: float,
                     out_q: Tensor = None,
                     out_k: Tensor = None):
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
