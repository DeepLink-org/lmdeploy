import torch
import deeplink_ext.cpp_extensions as ext
from torch import Tensor
from typing import Dict


def fill_kv_cache(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
                  v_caches: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                  kv_seq_length: Tensor, max_q_seq_length: int,
                  block_offsets: Tensor, **kwargs: Dict):
    """fill key/value state to cache for paged attention."""
    assert "kv_start_indices" in kwargs.keys()
    dest_index_copy_kv(k_states, kwargs["kv_start_indices"], k_caches)
    dest_index_copy_kv(v_states, kwargs["kv_start_indices"], v_caches)


def dest_index_copy_kv(states, dest_loc, caches):
    block_num, block_size, head, dim = caches.size()
    caches_tmp = caches.view(block_num * block_size, head, dim)
    ext.dest_index_copy_kv(states, dest_loc, caches_tmp)
    caches[:] = caches_tmp.view(block_num, block_size, head, dim)
