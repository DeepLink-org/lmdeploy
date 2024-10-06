# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import nn

from ..rotary_embedding import RotaryEmbeddingImpl

def _rotary_embedding_fwd(position_ids: torch.Tensor,
                          inv_freq: torch.Tensor,
                          scaling_factor: float,
                          mscale: float = None,
                          dtype: torch.dtype = None,
                          device_type: torch.device = None):
    """rotary embedding forward."""
    if dtype is None:
        dtype = torch.float16
    if device_type is None:
        device_type = 'cuda'
    position_ids = position_ids.float() / scaling_factor
    inv_freq_expanded = inv_freq[None, :,
                                 None].float().expand(position_ids.shape[0],
                                                      -1, 1)
    position_ids_expanded = position_ids[:, None, :]
    # Force float32 since bfloat16 loses precision on long contexts
    # See https://github.com/huggingface/transformers/pull/29285
    device_type = device_type if isinstance(
        device_type, str) and device_type != 'mps' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float()
                 @ position_ids_expanded.float()).transpose(1, 2)
        emb = freqs.repeat(1, 1, 2)
        cos = emb.cos()
        sin = emb.sin()

        if mscale is not None:
            cos = cos * mscale
            sin = sin * mscale

    return cos.to(dtype=dtype), sin.to(dtype=dtype)


class RotaryEmbeddingImpl(RotaryEmbeddingImpl, nn.Module):
    """base rotary embedding."""

    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        device_type = x.device.type
        dtype = x.dtype
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _rotary_embedding_fwd(position_ids,
                                     self.inv_freq,
                                     scaling_factor=self.scaling_factor,
                                     dtype=dtype,
                                     device_type=device_type)

class CambRotaryEmbeddingBuilder(RotaryEmbeddingBuilder):
    """rotary embedding builder."""

    @staticmethod
    def build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        yarn_params: YarnParameters = None,
        longrope_params: LongRoPEScalingParameters = None,
        llama3_params: Llama3Parameters = None,
        emb_type: RopeType = RopeType.Default,
    ):
        """build."""
        if emb_type in (RopeType.Default, RopeType.LinearScaling):
            return RotaryEmbeddingImpl(dim, base, scaling_factor)
        elif emb_type == RopeType.DynamicNTKScaling:
            return LlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings)
        elif emb_type == RopeType.Llama3:
            return Llama3RotaryEmbeddingImpl(dim, base, scaling_factor,
                                             llama3_params.low_freq_factor,
                                             llama3_params.high_freq_factor,
                                             max_position_embeddings)
        elif emb_type == RopeType.Yarn:
            return YarnRotaryEmbeddingImpl(dim,
                                           base,
                                           scaling_factor,
                                           max_position_embeddings,
                                           yarn_params=yarn_params)
        elif emb_type == RopeType.LongRoPEScaling:
            return LongRoPEScalingRotaryEmbeddingImpl(
                dim,
                base,
                max_position_embeddings=max_position_embeddings,
                longrope_params=longrope_params,
            )
        else:
            raise NotImplementedError(
                f'Unsupported embedding type: {emb_type}')

