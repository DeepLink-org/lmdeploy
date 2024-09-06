# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence
from dataclasses import dataclass

from torch import Tensor, IntTensor

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata

import torch
import infer_ext.ops as ext_ops

@dataclass
class AscendAttentionMetadata(AttentionMetadata):
    kv_start_indices: Optional[Tensor] = None
    block_size: int = 64
    attention_mask: Sequence[Tensor] = tuple()
    is_unpaged_prefill: Optional[bool] = None
    q_seqlens_int: IntTensor = None
    kv_seqlens_int: IntTensor = None
    kv_start_indices_1d: IntTensor = None
    block_offsets_int: IntTensor = None
    block_offsets_1d_int: IntTensor = None
    new_attn_mask: Tensor = None

class AscendAttentionImpl(AttentionImpl[AscendAttentionMetadata]):
    """ascend attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            v_head_size,
            alibi_scale,
            sliding_window,
            **kwargs,
        )

        from lmdeploy.pytorch.kernels.ascend import (fill_kv_cache,
                                                     paged_attention_fwd)
        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd
        self.block_size = 128

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        attn_metadata: AscendAttentionMetadata,
        inplace: bool = True,
    ) -> Tensor:
        """forward."""

        k_cache = k_cache.view(-1, self.block_size, self.num_kv_heads, self.v_head_size)
        v_cache = v_cache.view(-1, self.block_size, self.num_kv_heads, self.v_head_size)
        
        k_cache, v_cache = ext_ops.fill_kv_cache(key, value, k_cache, v_cache, attn_metadata.kv_start_indices_1d)

        if attn_metadata.is_decoding:
            block_offsets = attn_metadata.block_offsets_int
            attn_output = torch.ops.atb.paged_attention_decode(query, k_cache, v_cache, block_offsets, attn_metadata.kv_seqlens_int, None, self.num_heads, self.num_kv_heads)
        else:
            query = query.view(-1, self.num_heads * self.v_head_size)
            key = key.view(-1, self.num_kv_heads * self.v_head_size)
            value = value.view(-1, self.num_kv_heads * self.v_head_size)
            attn_output = torch.ops.atb.context_attention(query, key, value, k_cache, v_cache, attn_metadata.kv_seqlens_int, attn_metadata.new_attn_mask, self.num_heads, self.num_kv_heads)
        return attn_output

class AscendAttentionBuilder(AttentionBuilder[AscendAttentionMetadata]):
    """ascend attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ) -> AscendAttentionImpl:
        """build."""
        return AscendAttentionImpl(num_heads,
                                   head_size,
                                   scale=scale,
                                   num_kv_heads=num_kv_heads,
                                   v_head_size=v_head_size,
                                   alibi_scale=alibi_scale,
                                   sliding_window=sliding_window,
                                   logical_softcapping=logical_softcapping,
                                   **kwargs)
