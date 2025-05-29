# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    expert_offset: int,
    num_experts: int,
    renormalize: bool,
    ep_size: int,
    rank: int,
):
    """dlinfer fused moe."""
    if ep_size == 1:
        return ext_ops.fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk,
                                 renormalize)
    return ext_ops.fused_moe_with_ep(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk,
                                     expert_offset, num_experts, renormalize, ep_size, rank)
