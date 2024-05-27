import torch
import deeplink_ext.cpp_extensions as ext
from torch import Tensor
from torch.autograd.profiler import record_function


@record_function("mark_fill_kv_cache")
@torch.no_grad()
def fill_kv_cache(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
                  v_caches: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                  kv_seq_length: Tensor, max_q_seq_length: int,
                  block_offsets: Tensor, kv_start_indices: Tensor):
    """fill key/value state to cache for paged attention."""
    dest_index_copy_kv(k_states, kv_start_indices, k_caches)
    dest_index_copy_kv(v_states, kv_start_indices, v_caches)


@torch.no_grad()
def dest_index_copy_kv(states, dest_loc, caches):
    block_num, block_size, head, dim = caches.size()
    caches_tmp = caches.view(block_num * block_size, head, dim)
    ext.dest_index_copy_kv(states, dest_loc, caches_tmp)
    caches[:] = caches_tmp.view(block_num, block_size, head, dim)
