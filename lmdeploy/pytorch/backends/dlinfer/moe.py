# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from lmdeploy.pytorch.kernels.dlinfer import fused_moe, moe_gating_topk_softmax
from lmdeploy.pytorch.kernels.dlinfer.fused_moe import fused_moe_ascend_w8a8

from ..moe import (FusedMoEBuilder, FusedMoEImpl, FusedMoEW8A8Builder, FusedMoEW8A8Impl, SoftmaxTopKBuilder,
                   SoftmaxTopKImpl)


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


class DlinferFusedMoEAscendW8A8Impl(FusedMoEW8A8Impl):
    """Fused moe w8a8 implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.ep_size = ep_size

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(
                -1, -2).contiguous(), gate_up_scale, down_scale
        return gate_up_weights, down_weights, gate_up_scale, down_scale

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
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        return fused_moe_ascend_w8a8(hidden_states, gate_up_weights, gate_up_scale, down_weights, down_scale,
                                     topk_weights, topk_ids, self.top_k, self.num_experts, self.ep_size,
                                     self.renormalize)


class DlinferFusedMoEAscendW8A8Builder(FusedMoEW8A8Builder):
    """Fused moe w8a8 builder."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              out_dtype: torch.dtype = torch.float16,
              quant_dtype: torch.dtype = torch.int8,
              ep_size: int = 1):
        """Build from mlp."""
        return DlinferFusedMoEAscendW8A8Impl(top_k=top_k,
                                             num_experts=num_experts,
                                             renormalize=renormalize,
                                             ep_size=ep_size)
