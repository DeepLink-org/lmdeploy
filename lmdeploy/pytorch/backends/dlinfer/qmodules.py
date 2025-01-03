# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.dlinfer.w8a8_kernels import \
    rms_norm_dynamic_quant
from lmdeploy.pytorch.models.q_modules import QTensor

from ..qmodules import (LinearW8A8Builder, LinearW8A8Impl, RMSNormW8A8Builder,
                        RMSNormW8A8Impl)


class DlinferRMSNormW8A8Impl(RMSNormW8A8Impl):
    """dlinfer RMS norm w8a8 implementation api."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        """forward."""
        if residual is not None:
            x = x + residual
            residual = x
        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            x, weight, self.eps)
        x = QTensor(hidden_states_quant, rms_scale)
        if residual is None:
            return x
        return x, residual


class DlinferRMSNormBuilder(RMSNormW8A8Builder):
    """dlinfer RMS norm w8a8 implementation builder."""

    @staticmethod
    def build(hidden_size: int, eps: float = 1e-6):
        """build."""
        return DlinferRMSNormW8A8Impl(hidden_size, eps)


class DlinferLinearW8A8Impl(LinearW8A8Impl):
    """dlinfer linear w8a8 implementation."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def update_weights(self,
                       weight: torch.Tensor,
                       scale: torch.Tensor,
                       bias: Optional[torch.Tensor] = None):
        """update weights."""
        if os.getenv('DLINER_LINEAR_USE_NN_LAYOUT', '0') == '1':
            weight = weight.data.t().contiguous()
        return weight, scale, bias

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        if isinstance(x, torch.Tensor):
            x = x.contiguous()
            import vllm
            input_quant, input_scale, _ = vllm._custom_ops.scaled_int8_quant(
                x, None)

        else:
            assert isinstance(x, QTensor)
            input_quant, input_scale = x.tensor, x.scale

        bs, seq_len, head_size = input_quant.size()
        import pdb
        pdb.set_trace()
        out = vllm._custom_ops.cutlass_scaled_mm(input_quant.view(
            -1, head_size),
                                                 weight,
                                                 scale_a=input_scale,
                                                 scale_b=scale,
                                                 out_dtype=x.dtype,
                                                 bias=bias)
        import pdb
        pdb.set_trace()

        out = out.view(bs, seq_len, -1)
        # print(input_quant.shape, weight.shape, out.shape, flush=True)

        if all_reduce:
            dist.all_reduce(out)
        # import pdb; pdb.set_trace()
        return out


class DlinferLinearW8A8Builder(LinearW8A8Builder):
    """dlinfer linear w8a8 implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None):
        """build."""
        return DlinferLinearW8A8Impl(in_features, out_features)
