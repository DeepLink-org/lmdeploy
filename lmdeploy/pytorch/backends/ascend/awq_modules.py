# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
import torch

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


class AwqLinearW4A16Impl(LinearW4A16Impl):
    """awq kernel linear."""

    # mask_list = (0xF0000000, 0x0F000000, 0x00F00000, 0x000F0000, 0x0000F000,
    #              0x00000F00, 0x000000F0, 0x0000000F)
    # ascend_qweight_mapping = (7, 3, 6, 2, 5, 1, 4, 0)
    # ascend_qweight_mapping = (0, 4, 1, 5, 2, 6, 3, 7)

    def __init__(self, in_features: int, out_features: int, w_bit: int,
                 group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size

    # def update_weights(self,
    #                    qweight: torch.Tensor,
    #                    scales: torch.Tensor,
    #                    qzeros: torch.Tensor,
    #                    bias: Optional[torch.Tensor] = None):
    #     """update weights."""
    #     ascend_qweight_shifts = [
    #         (item - index) * 4 for index, item in enumerate(self.ascend_qweight_mapping)
    #     ]
    #     ascend_qweight = torch.zeros_like(qweight)
    #     qweight = torch.bitwise_xor(qweight, 0x88888888)
    #     for i in range(32 // self.w_bit):
    #         if ascend_qweight_shifts[i] > 0:
    #             ascend_qweight |= torch.bitwise_and(
    #                 qweight << ascend_qweight_shifts[i],
    #                 self.mask_list[i])
    #         elif ascend_qweight_shifts[i] < 0:
    #             ascend_qweight |= torch.bitwise_and(
    #                 qweight >> -ascend_qweight_shifts[i],
    #                 self.mask_list[i])
    #         else:
    #             ascend_qweight |= torch.bitwise_and(qweight, self.mask_list[i])

    #     shifts = torch.arange(0, 32, self.w_bit, device=qzeros.device)
    #     shifts = shifts.view(2, 4).t().flatten()
    #     izero = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    #     izero = izero.view(izero.shape[0], -1)
    #     ascend_qzero = (8 - torch.bitwise_and(izero, (2**self.w_bit) - 1)).to(
    #         torch.float16)

    #     del qweight, qzeros
    #     return ascend_qweight, scales, ascend_qzero, bias

    def forward(self,
                x,
                qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        out = ext_ops.weight_quant_matmul(
            x.squeeze(0),
            qweight,
            scales,
            offset=qzeros,
            bias=bias,
            group_size=self.group_size).unsqueeze(0)
        return out


class AwqLinearW4A16Builder(LinearW4A16Builder):
    """awq linear builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        return AwqLinearW4A16Impl(in_features, out_features, w_bit, group_size)


# grep .so w4a16
