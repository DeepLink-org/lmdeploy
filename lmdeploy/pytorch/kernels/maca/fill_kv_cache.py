# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor
from torch import Tensor
import infer_ext.ops as ext_ops

def fill_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
    key_caches: Tensor,
    value_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
    context: None
):
    # x = 32 // k_caches.element_size()
    # k_caches shape: [block_num, kv_head_num, head_size // x, block_size, x]
    # v_caches shape: [block_num, kv_head_num, block_size, head_size]
    ext_ops.fill_kv_cache(key_states, value_states,
                          key_caches, value_caches,
                          context.kv_start_indices)
