import torch
from torch import Tensor
from torch import nn

import vllm._C as vllm_ops


def fused_moe(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              topk: int,
              expert_offset: int = 0,
              num_experts: int = None,
              renormalize: bool = False) -> torch.Tensor:
    pass

    return
