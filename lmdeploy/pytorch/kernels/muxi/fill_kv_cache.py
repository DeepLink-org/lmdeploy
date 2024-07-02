import torch
from torch import Tensor
from typing import Dict

import vllm._C as vllm_ops

def fill_kv_cache(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
                  v_caches: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                  kv_seq_length: Tensor, max_q_seq_length: int,
                  block_offsets: Tensor, **kwargs: Dict):
    """fill key/value state to cache for paged attention."""
    assert "kv_start_indices" in kwargs.keys()

    _, head, dim = k_states.size()
    block_num, block_size_head_dim = v_caches.size()
    block_size = block_size_head_dim // (head * dim)
    x = 32 // k_caches.element_size()

    vllm_ops.cache_ops.reshape_and_cache_new(
        k_states,
        v_states,
        k_caches,
        v_caches,
        kwargs["kv_start_indices"],
        'auto',
    )
    lm_v_caches = v_caches.transpose(2, 1).contiguous()
    dest_index_copy_kv(v_states, kwargs["kv_start_indices"], lm_v_caches)
    lm_v_caches = lm_v_caches.transpose(2, 1).contiguous()
    v_caches[:] = lm_v_caches
    return


def dest_index_copy_kv(states, dest_loc, caches):
    block_num, block_size, head, dim = caches.size()
    caches_tmp = caches.view(block_num * block_size, head, dim)
    caches_tmp[dest_loc] = states
    caches[:] = caches_tmp.view(block_num, block_size, head, dim)
