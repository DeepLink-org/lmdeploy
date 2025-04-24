# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Optional

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.kernels.dlinfer import linear

from ..linear import LinearBuilder, LinearImpl
from torch import distributed as torch_dist
# from ..default.linear import _reduce_scatter_input


def _reduce_scatter_input(out: torch.Tensor, rank: int, tp_sizes: List[int]):
    """reduce scatter."""
    out = out.transpose(0, -2)
    if not out.is_contiguous():
        out = out.contiguous()
    outs = out.split(tp_sizes, 0)
    out = outs[rank]
    outs = list(outs)
    # dist.reduce_scatter(out, outs)
    # reduce
    for tensor in outs:
        dist.all_reduce(tensor,  op=dist.ReduceOp.SUM)
    
    # scatter
    combined = torch.cat(outs)
    world_size = torch_dist.get_world_size() 
    chunk_size = combined.shape[0]  // world_size 
    start = rank * chunk_size 
    end = start + chunk_size if rank != world_size - 1 else combined.shape[0] 
    
    # copy
    out.copy_(combined[start:end])

    out = out.transpose(0, -2)
    return out

class DlinferLinearImpl(LinearImpl):
    """Dlinfer linear implementation api."""

    def update_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """update weights."""
        if os.getenv('DLINER_LINEAR_USE_NN_LAYOUT', '0') == '1':
            weight = weight.data.t().contiguous()
        return weight, bias

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        out = linear(x, weight, bias, False)
        if all_reduce:
            if scatter_size is not None:
                out = _reduce_scatter_input(out, rank, scatter_size)
            else:
                dist.all_reduce(out)
        return out


class DlinferLinearBuilder(LinearBuilder):
    """Dlinfer linear implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return DlinferLinearImpl()
