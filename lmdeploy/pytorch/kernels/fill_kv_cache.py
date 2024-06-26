import torch
##import deeplink_ext.cpp_extensions as ext
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
    #ext.dest_index_copy_kv(states, dest_loc, caches_tmp)
    caches_tmp[dest_loc] = states
    caches[:] = caches_tmp.view(block_num, block_size, head, dim)

#def _div_up(a, b):
#    return (a + b - 1) // b
#
#def fill_kv_cache(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
#                  v_caches: Tensor,
#                  q_start_loc: Tensor,
#                  q_seq_length: Tensor,
#                  kv_seq_length: Tensor,
#                  max_q_seq_length: int,
#                  block_offsets: Tensor,
#                  **kwargs: Dict):
##def gt(self, k_states, v_states, k_caches, v_caches, seq_lens,
##       history_lens, block_offsets, block_size):
#    #block_num, block_size, head, dim = k_caches.size()
#    #seq_lens = q_seq_length
#    #history_lens = kv_seq_length
#    #batch_size = len(seq_lens)
#
#    #block_num, block_size, head, dim = k_caches.size()
#    #batch_size = len(seq_lens)
#    #batch_k = []
#    #batch_v = []
#    #for i in range(batch_size):
#    #    block_id = _div_up(kv_seq_length[i], block_size) - 1
#    #    offset_id = (kv_seq_length[i] - 1) % block_size
#    #    k_caches[block_offsets[i]][block_id, offset_id] = k_states[i]
#    #    import pdb;pdb.set_trace()
#    #    k_caches[block_offsets[i]].reshape(-1, head, dim)[kv_seq_length[2] - 1]
#    #    offsets = block_offsets[i]
#    #    tmp_k = key_cache[offsets].view(-1, head, dim)
#    #    tmp_k = tmp_k[:kv_seqlens[i] + 1]
#    #    batch_k.append(tmp_k)
#
#    #    tmp_v = value_cache[offsets].view(-1, head, dim)
#    #    tmp_v = tmp_v[:kv_seqlens[i] + 1]
#    #    batch_v.append(tmp_v)
#    #batched_k = torch.stack(batch_k, 0)
#    #batched_v = torch.stack(batch_v, 0)
#    #bias = _make_bias(q_seqlens, kv_seqlens)
#    #attn_output.copy_(_naive_attention(query_states.unsqueeze(1), batched_k, batched_v, bias))
#    ##import pdb;pdb.set_trace()
#    ##pass
#
#    block_num, block_size, head, dim = k_caches.size()
#    seq_lens = q_seq_length
#    history_lens = kv_seq_length
#    batch_size = len(seq_lens)
#    #k_caches = k_caches.clone()
#    #v_caches = v_caches.clone()
#    splited_k_states = k_states.split(seq_lens.tolist())
#    splited_v_states = v_states.split(seq_lens.tolist())
#    for bidx in range(batch_size):
#        k_state = k_states[bidx]
#        v_state = v_states[bidx]
#        h_len = history_lens[bidx] - 1
#        b_offs = block_offsets[bidx]
#        block_id = _div_up(h_len + 1, block_size) - 1
#        fill_start = h_len % block_size
#        fill_size = min(block_size - fill_start, k_state.size(0))
#        while True:
#            boff = b_offs[block_id]
#            tmp_ks = k_state[:fill_size]
#            tmp_vs = v_state[:fill_size]
#            fill_end = fill_start + fill_size
#            k_caches[boff, fill_start:fill_end] = tmp_ks
#            v_caches[boff, fill_start:fill_end] = tmp_vs
#            k_state = k_state[fill_size:]
#            v_state = v_state[fill_size:]
#            block_id += 1
#            fill_start = 0
#            fill_size = min(block_size, k_state.size(0))
#            if fill_size == 0:
#                break
#    
#    #yield k_caches, v_caches


