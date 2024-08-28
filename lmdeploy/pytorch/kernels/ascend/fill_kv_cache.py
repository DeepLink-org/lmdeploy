# Copyright (c) OpenMMLab. All rights reserved.
# import dlinfer.ops as ext_ops
from torch import Tensor


# def fill_kv_cache(
#     key_states: Tensor,
#     value_states: Tensor,
#     key_caches: Tensor,
#     value_caches: Tensor,
#     q_start_loc: Tensor,
#     q_seq_length: Tensor,
#     kv_seq_length: Tensor,
#     max_q_seq_length: int,
#     block_offsets: Tensor,
#     context: None,
# ):
#     """fill key/value state to cache for paged attention."""
#     ext_ops.fill_kv_cache(key_states, value_states, key_caches, value_caches,
#                           context.kv_start_indices)


def fill_kv_cache(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
                  v_caches: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                  kv_seq_length: Tensor, max_q_seq_length: int,
                  block_offsets: Tensor, context: None):
    """fill key/value state to cache for paged attention."""
    _, block_size, _, _ = k_caches.size()
    for i in range(q_start_loc.size(0)):
        history_seqlen = kv_seq_length[i] - q_seq_length[i]
        block_idx = history_seqlen // block_size
        block_id = block_offsets[i][block_idx]
        token_loc = history_seqlen % block_size
        for offset in range(q_seq_length[i]):
            state_idx = q_start_loc[i] + offset
            k_caches[block_id][token_loc] = k_states[state_idx]
            v_caches[block_id][token_loc] = v_states[state_idx]
            if offset == q_seq_length[i] - 1:
                break
            token_loc = (token_loc + 1) % block_size
            block_idx = block_idx if token_loc else block_idx + 1
            block_id = block_id if token_loc else block_offsets[i][block_idx]
