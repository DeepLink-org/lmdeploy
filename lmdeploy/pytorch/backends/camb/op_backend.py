# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


class CambOpsBackend(DefaultOpsBackend):
    """ascend layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'camb'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get ascend layer builder."""
        if layer_type == OpType.Attention:
            from .attention import CambAttentionBuilder
            return CambAttentionBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from .apply_rotary_emb import CambApplyRotaryEmbBuilder
            return CambApplyRotaryEmbBuilder
        elif layer_type == OpType.RMSNorm:
            from .norm import CambRMSNormBuilder
            return CambRMSNormBuilder
        #elif layer_type == OpType.RotaryEmbedding:
        #    from .rotary_embedding import CambRotaryEmbeddingBuilder
        #    return CambRotaryEmbeddingBuilder
        else:
            logger.debug(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        from .attention import CambAttentionMetadata
        return CambAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            #block_size,
            num_heads,
            block_size,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            #block_size,
            num_heads,
            block_size,
            head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        kv_start_indices, attention_mask = [], []
        #_, block_size, _, _ = step_context.kv_caches[0][0].shape
        _, _, block_size, _ = step_context.kv_caches[0][0].shape
        device = step_context.block_offsets.device
        batch_size = step_context.q_start_loc.shape[0]

        is_unpaged_prefill = False
        q_start_loc = step_context.q_start_loc
        q_seqlens = step_context.q_seqlens
        kv_seqlens = step_context.kv_seqlens.to(torch.int32)
        max_q_seq_len = torch.max(q_seqlens).cpu().item()
        max_kv_seq_len = torch.max(kv_seqlens).cpu().item()

        cu_seqlens = torch.zeros(batch_size+1, dtype=torch.int32, device=device)
        cu_seqlens[:-1] = step_context.q_start_loc
        cu_seqlens[-1] = step_context.q_seqlens.sum()
        cu_seqlens_list = cu_seqlens.tolist()

        if not step_context.is_decoding:
            cos_sin_ids = step_context.position_ids[0]
        else:
            cos_sin_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)

        if not step_context.is_decoding:
            is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())

        for i in range(batch_size):
            q_seq_len = int(step_context.q_seqlens[i])
            kv_seq_len = int(step_context.kv_seqlens[i])
            history_length = kv_seq_len - q_seq_len
            block_idx = history_length // block_size
            block_loc = step_context.block_offsets[i][block_idx]
            token_loc = history_length % block_size
            for j in range(q_seq_len):
                kv_start_indices.append(block_loc * block_size + token_loc)
                if j == q_seq_len - 1:
                    break
                token_loc = (token_loc + 1) % block_size
                block_idx = block_idx if token_loc else block_idx + 1
                block_loc = step_context.block_offsets[i][block_idx]
        kv_start_indices = torch.tensor(kv_start_indices, device=device)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=None,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            cu_seqlens=cu_seqlens,
            cos_sin_ids=cos_sin_ids,
        )

        step_context.attn_metadata = attn_metadata
        return step_context

