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
    num_experts: int,
    ep_size: int,
    renormalize: bool,
):
    """Dlinfer fused moe."""
    if ep_size != 1:
        return ext_ops.fused_moe_with_alltoall(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids,
                                               topk, num_experts, ep_size, renormalize)
    return ext_ops.fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk, renormalize)


def fused_moe_ascend_w8a8(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    gate_up_scale: Tensor,
    down_weights: Tensor,
    down_scale: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    num_experts: int,
    ep_size: int,
    renormalize: bool,
):
    """Dlinfer fused moe."""
    return ext_ops.fused_moe_ascend_w8a8(hidden_states, gate_up_weights, gate_up_scale, down_weights, down_scale,
                                         topk_weights, topk_ids, topk, num_experts, ep_size, renormalize)
