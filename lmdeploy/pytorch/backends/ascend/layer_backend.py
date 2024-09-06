# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..base import LayerType
from ..default import DefaultLayersBackend

logger = get_logger('lmdeploy')

import torch._dynamo as dynamo

class AscendLayersBackend(DefaultLayersBackend):
    """ascend layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        raise 'ascend'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: LayerType):
        """get ascend layer builder."""
        if layer_type == LayerType.Attention:
            from .attention import AscendAttentionBuilder
            return AscendAttentionBuilder
        elif layer_type == LayerType.ApplyRotaryEmb:
            from .apply_rotary_emb import AscendApplyRotaryEmbBuilder
            return AscendApplyRotaryEmbBuilder
        elif layer_type == LayerType.RMSNorm:
            from .norm import AscendRMSNormBuilder
            return AscendRMSNormBuilder
        elif layer_type == LayerType.RotaryEmbedding:
            from .rotary_embedding import AscendRotaryEmbeddingBuilder
            return AscendRotaryEmbeddingBuilder
        else:
            logger.debug(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        from .attention import AscendAttentionMetadata
        return AscendAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads * head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads * head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        kv_start_indices, attention_mask = [], []
        _, block_size, _ = step_context.kv_caches[0][0].shape
        device = step_context.block_offsets.device
        for i in range(step_context.q_start_loc.size(0)):
            q_seq_len = int(step_context.q_seqlens[i])
            kv_seq_len = int(step_context.kv_seqlens[i])
            # single_attention_mask = torch.logical_not(
            #     torch.tril(
            #         torch.ones(q_seq_len, kv_seq_len,
            #                    dtype=torch.bool, device=device),
            #         diagonal=kv_seq_len - q_seq_len,
            #     ))
            # attention_mask.append(single_attention_mask)
            history_length = kv_seq_len - q_seq_len
            block_idx = history_length // block_size
            block_loc = step_context.block_offsets[i][block_idx]
            token_loc = history_length % block_size
            for _ in range(q_seq_len):
                kv_start_indices.append([block_loc * block_size + token_loc])
                if _ == q_seq_len - 1:
                    break
                token_loc = (token_loc + 1) % block_size
                block_idx = block_idx if token_loc else block_idx + 1
                block_loc = step_context.block_offsets[i][block_idx]
        kv_start_indices = torch.tensor(
            kv_start_indices, device=device)

        max_seq_len = step_context.attention_mask.shape[1]
        if not step_context.is_decoding:
            new_attn_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.float16, device='npu'), diagonal=1)
        else:
            new_attn_mask = None
        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets.contiguous(),
            q_start_loc=step_context.q_start_loc,
            q_seqlens=step_context.q_seqlens,
            kv_seqlens=step_context.kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=attention_mask,
            q_seqlens_int=step_context.q_seqlens.to(torch.int32),
            kv_seqlens_int=step_context.kv_seqlens.to(torch.int32),
            kv_start_indices_1d=kv_start_indices.flatten().to(torch.int32),
            block_offsets_int=step_context.block_offsets.to(torch.int32).contiguous(),
            block_offsets_1d_int=step_context.block_offsets.flatten().to(torch.int32).contiguous(),
            new_attn_mask=new_attn_mask
        )
        dynamo.mark_dynamic(attn_metadata.block_offsets_int, [0, 1])
        if not step_context.is_decoding:
            attn_metadata.is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())
        step_context.attn_metadata = attn_metadata
        return step_context
