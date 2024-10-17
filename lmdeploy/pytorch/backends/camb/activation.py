# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.kernels.camb.activation import silu_and_mul

from ..activation import (GeluAndMulBuilder, GeluAndMulImpl, SiluAndMulBuilder,
                          SiluAndMulImpl)

class CambSiluAndMulImpl(SiluAndMulImpl):
    """silu + multiple fused implementation."""

    def forward(self, x):
        """forward."""
        return silu_and_mul(x)

class CambSiluAndMulBuilder(SiluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(inplace: bool = False):
        """build."""
        return CambSiluAndMulImpl()
