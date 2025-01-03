# Copyright (c) OpenMMLab. All rights reserved.
import torch


def matmul_kernel_dynamic_quant(a,
                                b,
                                rms_scale,
                                linear_scale,
                                residual=None,
                                bias=None,
                                output_dtype=torch.float16):
    """This function performs matrix multiplication with dynamic quantization.

    It takes two input tensors `a` and `b`, scales them with `rms_scale` and
    `linear_scale`, and optionally adds a `residual` tensor and a `bias`. The
    output is returned in the specified `output_dtype`.
    """
    return


def per_token_quant_int8(x, eps):
    """Function to perform per-token quantization on an input tensor `x`.

    It converts the tensor values into signed 8-bit integers and returns the
    quantized tensor along with the scaling factor used for quantization.
    """

    return


def rms_norm_dynamic_quant(x, w, eps):
    """Performs RMS normalization with dynamic quantization.

    The function reshapes the input tensor `x`, creates an empty tensor `y`
    with the same shape as `x`, and calculates RMS normalization on the
    reshaped `x` using a Triton kernel `_rms_norm_fwd_fused_dynamic_symmetric`.
    """
    return
