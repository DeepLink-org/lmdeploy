# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from lmdeploy.pytorch.kernels.dlinfer import fused_moe, moe_gating_topk_softmax

from ..moe import FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder, SoftmaxTopKImpl


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """Dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(x, self.top_k)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """Dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """Dlinfer fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.ep_size = ep_size

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(-1, -2).contiguous()
        return gate_up_weights, down_weights

    def support_ep(self):
        """Support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        expert_per_rank = (self.num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, self.num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, self.top_k,
                         self.num_experts, self.ep_size, self.renormalize)


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """Dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        """Build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize, ep_size=ep_size)
