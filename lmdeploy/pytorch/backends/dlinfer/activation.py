# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.dlinfer import silu_and_mul

from ..activation import (GeluAndMulBuilder, GeluAndMulImpl, SiluAndMulBuilder,
                          SiluAndMulImpl)

class DlinferSiluAndMulImpl(SiluAndMulImpl):
    """silu + multiple fused implementation."""

    def forward(self,
            x: torch.Tensor):
        """forward."""
        return silu_and_mul(x)

class DlinferSiluAndMulBuilder(SiluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(inplace: bool = False):
        """build."""
        return DlinferSiluAndMulImpl()
