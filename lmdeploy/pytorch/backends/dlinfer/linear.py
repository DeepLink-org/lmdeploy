# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Optional

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.kernels.dlinfer import linear

from ..linear import LinearBuilder, LinearImpl


def _gather_input(x: torch.Tensor, tp_sizes: List[int]):
    """gather input."""
    shape0 = x.shape[:-2]
    shape1 = x.shape[-1:]
    max_tp_size = max(tp_sizes)
    if x.shape[-2] < max_tp_size:
        pad = torch.zeros(((max_tp_size, ) + shape1), device=x.device, dtype=x.dtype)
        pad = pad[x.shape[-2]:]
        x = torch.cat((x, pad), dim=0)
    shapes = [(max_tp_size, ) + shape1 for _ in tp_sizes]
    # shapes = [shape0 + (size, ) + shape1 for size in tp_sizes]
    new_x = [x.new_empty(shape) for shape in shapes]
    dist.all_gather(new_x, x)
    x = torch.cat(new_x, dim=-2)
    return x


def _reduce_scatter_input(out: torch.Tensor, rank: int, tp_sizes: List[int]):
    """reduce scatter."""
    out = out.transpose(0, -2)
    if not out.is_contiguous():
        out = out.contiguous()
    max_tp_size = max(tp_sizes)
    outs = out.split([max_tp_size] * len(tp_sizes), 0)
    out = outs[rank]
    outs = list(outs)
    # dist.reduce_scatter(out, outs)
    for tensor in outs:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    out.copy_(outs[rank])

    out = out[:tp_sizes[rank]].transpose(0, -2)
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
                dp_gather: bool = False,
                all_reduce: bool = False,
                tp_size: List[int] = None):
        """forward."""
        # return linear(x, weight, bias, dp_gather, all_reduce, rank, tp_size)
        if dp_gather:
            x = _gather_input(x.squeeze(0), tp_size)
        _, rank = dist.get_tp_world_rank()
        out = linear(x, weight, bias, dp_gather, all_reduce, rank, tp_size)
        if all_reduce and tp_size is not None:
            # out = out[:tp_size[rank]].unsqueeze(0)
            out = out.unsqueeze(0)
        return out
        # if all_reduce:
        #     if tp_size is not None:
        #         out = _reduce_scatter_input(out, rank, tp_size)
        #     else:
        #         dist.all_reduce(out)
        # return out


class DlinferLinearBuilder(LinearBuilder):
    """Dlinfer linear implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return DlinferLinearImpl()
