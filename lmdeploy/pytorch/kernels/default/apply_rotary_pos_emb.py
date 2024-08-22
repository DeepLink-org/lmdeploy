import torch
from torch import Tensor



def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor = None,
    position_ids_1d: Tensor = None,
    q_embed=None,
    k_embed=None,
    context=None,
):
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    bs, seq_len, head, dim = query_states.shape
    numKeyValueHeads = key_states.shape[2]
    cos = cos[position_ids_1d].view(1, seq_len, 1, -1)[..., :dim//2]
    sin = sin[position_ids_1d].view(1, seq_len, 1, -1)[..., :dim//2]
    q1, q2 = query_states.split(query_states.size(-1) // 2, dim=-1)
    k1, k2 = key_states.split(key_states.size(-1) // 2, dim=-1)
    q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rotated.view(bs*seq_len, head, dim), k_rotated.view(bs*seq_len, numKeyValueHeads, dim)
