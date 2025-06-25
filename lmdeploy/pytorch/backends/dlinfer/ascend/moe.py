# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from ...moe import FusedMoEBuilder, FusedMoEImpl
from ..token_dispatcher import TokenDispatcherBuilder
from lmdeploy.pytorch.distributed import get_ep_world_rank, get_dist_manager

import dlinfer.ops as ext_ops


class AscendFusedMoEImpl(FusedMoEImpl):
    """ascend fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False, hidden_dim: int = 0, dtype: Optional[torch.dtype] = None, enable_ep: bool = False):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        dist_ctx = get_dist_manager().current_context()
        ep_size, ep_rank = get_ep_world_rank()
        self.token_dispatcher = TokenDispatcherBuilder.build(
            group=dist_ctx.ep_gpu_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=dtype,
        )

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(-1, -2).contiguous()
        return gate_up_weights, down_weights

    def support_ep(self):
        """support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        recv_hidden_states, recv_topk_ids, recv_topk_weights, tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            expert_list,
        )
        out_states = ext_ops.fused_moe(recv_hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, self.top_k,
                         self.renormalize)
        out_states = self.token_dispatcher.combine(out_states)
        return out_states        # if self.renormalize:
        #     topk_weights = ext_ops.renormalize(topk_weights)
        # expanded_hidden_states, expanded_row_idx, group = ext_ops.moe_init_routing(hidden_states, topk_ids, self.num_experts)
        # expert_tokens = torch.bincount(expanded_row_idx, minlength=self.num_experts)
        # ep_size, ep_rank = get_ep_world_rank()
        # scatter_sizes = expert_tokens.view(ep_size, -1).sum(-1)
        # gather_sizes = torch.empty_like(scatter_sizes)
        # dist.all_to_all_single(gather_sizes, scatter_sizes)
        # scatter_size_list = scatter_sizes.cpu().tolist()
        # gather_size_list = gather_sizes.cpu().tolist()
        # expanded_row_idx = expanded_row_idx % (self.num_experts // ep_size)
        # hidden_states = dist.all_to_all(hidden_states, 0, 0,
        #                                     scatter_size_list,
        #                                     gather_size_list)
        # local_expert_idx = dist.all_to_all(expanded_expert_idx, 0, 0,
        #                                        scatter_size_list,
        #                                        gather_size_list)

        # up_proj = ext_ops.grouped_matmul(expanded_hidden_states, gate_up_weights, group, 2)
        # gate_cache = ext_ops.silu_and_mul(up_proj, -1)
        # down_proj = ext_ops.grouped_matmul(gate_cache, down_weights, group, 2)
        # moe_out = ext_ops.moe_token_unpermute(down_proj, expanded_row_idx, topk_weights)
        # return moe_out
    
    # def fused_moe_build(self):



class AscendFusedMoEBuilder(FusedMoEBuilder):
    """ascend fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False, hidden_dim: int = 0, dtype: Optional[torch.dtype] = None, enable_ep: bool = False):
        """build from mlp."""
        return AscendFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize)
