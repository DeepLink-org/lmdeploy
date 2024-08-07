# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import dlinfer.ops as ext_ops
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from lmdeploy.pytorch.kernels.ascend.fused_rotary_emb import fused_rotary_emb
from lmdeploy.pytorch.kernels.ascend.paged_attention_fwd import paged_attention_fwd

from ..kernels import fill_kv_cache
from ..weight_loader.dist_utils import (colwise_split_parallelize_linear,
                                        rowwise_parallelize_linear)

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


# flake8: noqa: F821


def get_range_ones(mask):
    mask = mask.tolist()[0]
    res_range = []
    inv_range = []
    count = 0
    inv_start = 0
    prev_diff = -1

    # get range for continous ones
    for i in range(len(mask)):
        count += mask[i]

        # handling ones
        # insert range at the end
        if i > 0 and mask[i] == 0 and mask[i-1] == 1:
            end = i - 1
            res_range.append([start, end])
        # handling zero
        # insert inv_range at the end
        if i > 0 and mask[i] == 1 and mask[i-1] == 0:
            inv_end = i - 1
            inv_range.append([inv_start, inv_end])
        # prepare for next range
        if i - count != prev_diff:
            start = i + 1
        else:
            inv_start = i + 1
        prev_diff = i - count

    # last range block
    if mask[-1] == 1:
        res_range.append([start, len(mask) - 1])
    else:
        inv_range.append([inv_start, len(mask) - 1])
    return res_range, inv_range


def merge_section_size(left_range, right_range):
    def get_len(range_elem):
        return range_elem[1] - range_elem[0] + 1

    res_size = []
    side_idx = []
    l_idx = r_idx = 0

    # merge sort for l&r array
    while l_idx < len(left_range) and r_idx < len(right_range):
        if left_range[l_idx] < right_range[r_idx]:
            l_len = get_len(left_range[l_idx])
            res_size.append(l_len)
            side_idx.append(0)
            l_idx += 1
        else:
            r_len = get_len(right_range[r_idx])
            res_size.append(r_len)
            side_idx.append(1)
            r_idx += 1

    # handle tailing data
    if l_idx < len(left_range):
        res_size.extend([get_len(elem) for elem in left_range[l_idx:]])
        side_idx.append(0)
    if r_idx < len(right_range):
        res_size.extend([get_len(elem) for elem in right_range[r_idx:]])
        side_idx.append(1)
    return res_size, side_idx


def handle_mask_range_split(in_data, lang_fn, vision_fn, context, stage_info=None):
    # split inputs for continous slice batch
    all_mask_size, side_idx = merge_section_size(
        context.vision_token_range, context.language_token_range)
    split_hidden_states = torch.split(in_data, all_mask_size, dim=1)

    # calculate and merge
    output_layer = []
    for i, elem in enumerate(split_hidden_states):
        # language part
        if side_idx[i] == 1:
            output_layer.append(lang_fn(elem))
        # vision part
        else:
            output_layer.append(vision_fn(elem))
    output_layer = torch.cat(output_layer, dim=1)
    return output_layer


def get_vision_expert_mask(
    token_type_ids: 'torch.LongTensor(B, L)'
) -> '[torch.BoolTensor(B, L), torch.BoolTensor(B, L)]':
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1]
                                 == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:]
                                                          == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class PatchedVisionExpertMLP(nn.Module):

    def forward(self, hidden_states: 'torch.Tensor(B, L, D)',
                token_type_ids: 'torch.LongTensor(B, L)'):
        context = self.context.context
        only_has_language = context.is_decoding
        if not context.is_decoding:
            # for embedding splitting
            if hasattr(context, 'vision_token_mask') and hasattr(
                    context, 'language_token_mask'):
                vision_token_mask = context.vision_token_mask
                language_token_mask = context.language_token_mask
                only_has_language = vision_token_mask.numel() == 0
            else:
                only_has_language = True

        if only_has_language:
            output = self.language_mlp(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            output[:, vision_token_mask, :] = self.vision_mlp(
                hidden_states[:, vision_token_mask, :])
            output[:, language_token_mask, :] = self.language_mlp(
                hidden_states[:, language_token_mask, :])
        return output


class PatchedVisionExpertMLPAscend(nn.Module):

    def forward(self, hidden_states: 'torch.Tensor(B, L, D)',
                token_type_ids: 'torch.LongTensor(B, L)'):
        context = self.context.context
        only_has_language = context.is_decoding
        if not context.is_decoding:
            # for embedding splitting
            if hasattr(context, 'vision_token_mask') and hasattr(
                    context, 'language_token_mask'):
                vision_token_mask = context.vision_token_mask
                language_token_mask = context.language_token_mask
                only_has_language = vision_token_mask.numel() == 0
            else:
                only_has_language = True

        if only_has_language:
            output = self.language_mlp(hidden_states)
        else:
            output = handle_mask_range_split(hidden_states,
                                             self.language_mlp,
                                             self.vision_mlp,
                                             self.context.context)
        return output


class PatchedVisionExpertAttention(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_multi_query_heads', num_heads)
        head_dim = self.config.hidden_size // num_heads
        sections = [
            self.config.hidden_size, num_kv_heads * head_dim,
            num_kv_heads * head_dim
        ]
        for name in [
                'vision_expert_query_key_value',
                'language_expert_query_key_value'
        ]:
            colwise_split_parallelize_linear(getattr(self, name),
                                             sections,
                                             loader,
                                             rank=rank,
                                             world_size=world_size,
                                             prefix=name)
        for name in ['vision_expert_dense', 'language_expert_dense']:
            rowwise_parallelize_linear(getattr(self, name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of Attention.forward.

        Add continuous batching support. Add paged attention support.
        """
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        q_seq_length_list = context.q_seq_length_list
        max_q_seq_length = context.max_q_seq_length
        num_heads = self.config.num_attention_heads // world_size
        num_kv_heads = getattr(self.config, 'num_multi_query_heads',
                               self.config.num_attention_heads) // world_size

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        hidden_size = num_heads * head_dim
        only_has_language = context.is_decoding
        if not context.is_decoding:
            # for embedding splitting
            if hasattr(context, 'vision_token_mask') and hasattr(
                    context, 'language_token_mask'):
                vision_token_mask = context.vision_token_mask
                language_token_mask = context.language_token_mask
                only_has_language = vision_token_mask.numel() == 0
            else:
                only_has_language = True

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            if only_has_language:
                mixed_raw_layer = self.language_expert_query_key_value(
                    hidden_states)
            else:
                shape = list(hidden_states.shape)
                shape[-1] = hidden_size + head_dim * num_kv_heads * 2
                mixed_raw_layer = torch.empty(shape,
                                              dtype=hidden_states.dtype,
                                              device=hidden_states.device)

                mixed_raw_layer[:,
                                vision_token_mask, :] = self.vision_expert_query_key_value(
                                    hidden_states[:, vision_token_mask, :])
                mixed_raw_layer[:,
                                language_token_mask, :] = self.language_expert_query_key_value(
                                    hidden_states[:, language_token_mask, :])
            query_states, key_states, value_states = torch.split(
                mixed_raw_layer, [
                    hidden_size, head_dim * num_kv_heads,
                    head_dim * num_kv_heads
                ],
                dim=-1)
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            scaling_factor = getattr(self.rotary_emb, 'scaling_factor', 1.0)
            inv_freq = self.rotary_emb.inv_freq

            query_states, key_states = fused_rotary_emb(
                query_states[None],
                key_states[None],
                position_ids[None],
                inv_freq=inv_freq,
                scaling_factor=scaling_factor,
                out_q=query_states[None],
                out_k=key_states[None])
            return query_states[0], key_states[0], value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)

        fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
            context=self.context.context
        )

        context_layer = query_states
        paged_attention_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            context_layer,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            q_seqlens_list=q_seq_length_list,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            context=self.context.context
        )
        context_layer = context_layer.reshape(*hidden_states.shape[:-1], -1)

        if only_has_language:
            attn_output = self.language_expert_dense(context_layer)
        else:
            ctx_shape = list(context_layer.shape)
            ctx_shape[-1] *= world_size
            attn_output = torch.empty(ctx_shape,
                                      dtype=hidden_states.dtype,
                                      device=hidden_states.device)

            attn_output[:, vision_token_mask, :] = self.vision_expert_dense(
                context_layer[:, vision_token_mask, :])
            attn_output[:,
                        language_token_mask, :] = self.language_expert_dense(
                            context_layer[:, language_token_mask, :])

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            world_size=world_size,
        )


class PatchedVisionExpertAttentionAscend(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_multi_query_heads', num_heads)
        head_dim = self.config.hidden_size // num_heads
        sections = [
            self.config.hidden_size, num_kv_heads * head_dim,
            num_kv_heads * head_dim
        ]
        for name in [
                'vision_expert_query_key_value',
                'language_expert_query_key_value'
        ]:
            colwise_split_parallelize_linear(getattr(self, name),
                                             sections,
                                             loader,
                                             rank=rank,
                                             world_size=world_size,
                                             prefix=name)
        for name in ['vision_expert_dense', 'language_expert_dense']:
            rowwise_parallelize_linear(getattr(self, name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of Attention.forward.

        Add continuous batching support. Add paged attention support.
        """
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        q_seq_length_list = context.q_seq_length_list
        max_q_seq_length = context.max_q_seq_length
        num_heads = self.config.num_attention_heads // world_size
        num_kv_heads = getattr(self.config, 'num_multi_query_heads',
                               self.config.num_attention_heads) // world_size

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        hidden_size = num_heads * head_dim
        only_has_language = context.is_decoding
        if not context.is_decoding:
            # for embedding splitting
            if hasattr(context, 'vision_token_mask') and hasattr(
                    context, 'language_token_mask'):
                vision_token_mask = context.vision_token_mask
                language_token_mask = context.language_token_mask
                only_has_language = vision_token_mask.numel() == 0
            else:
                only_has_language = True

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            if only_has_language:
                mixed_raw_layer = self.language_expert_query_key_value(
                    hidden_states)
            else:
                mixed_raw_layer = handle_mask_range_split(hidden_states,
                                                          self.language_expert_query_key_value,
                                                          self.vision_expert_query_key_value,
                                                          self.context.context)

            query_states, key_states, value_states = torch.split(
                mixed_raw_layer, [
                    hidden_size, head_dim * num_kv_heads,
                    head_dim * num_kv_heads
                ],
                dim=-1)
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            scaling_factor = getattr(self.rotary_emb, 'scaling_factor', 1.0)
            inv_freq = self.rotary_emb.inv_freq

            q = query_states[None]
            k = key_states[None]
            batch, seqlen, _, _ = q.shape

            pos_id = position_ids[None].squeeze(0).unsqueeze(-1)
            pos_freq = pos_id / scaling_factor * inv_freq
            cos = (torch.cos(pos_freq).view(batch, seqlen, 1,
                                            -1).repeat(1, 1, 1,
                                                    2).to(q.dtype))
            sin = (torch.sin(pos_freq).view(batch, seqlen, 1,
                                            -1).repeat(1, 1, 1,
                                                    2).to(q.dtype))
            ext_ops.apply_rotary_pos_emb(q, k,
                                        cos, sin, None, None)
            return q[0], k[0], value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)

        ext_ops.fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            self.context.context.kv_start_indices
        )

        context_layer = query_states
        paged_attention_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            context_layer,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            context=self.context.context
        )
        context_layer = context_layer.reshape(*hidden_states.shape[:-1], -1)

        if only_has_language:
            attn_output = self.language_expert_dense(context_layer)
        else:
            attn_output = handle_mask_range_split(context_layer,
                                                  self.language_expert_dense,
                                                  self.vision_expert_dense,
                                                  self.context.context)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            world_size=world_size,
        )


class PatchedCogVLMModel(nn.Module):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # not allow for inputs_embeds, because we want to process image feature
        assert input_ids is not None
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = _get_cogvlm_position_ids(context)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            # multi-modality
            if len(context.token_type_range) == 1:
                inputs_embeds[:,
                          context.token_type_range[0][0] : context.token_type_range[0][1] + 1, :] = vision_embeddings.to(
                              inputs_embeds)
            else:        
                inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=None,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_value,
            hidden_states=None,
            attentions=None,
        )


def build_position_ids(
        x: 'torch.BoolTensor(B, L)') -> 'torch.LongTensor(B, L)':
    tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (
        tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (
        tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (
        (tmp[:, 1:] == VISION_TOKEN_TYPE) &
        (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y


def _get_cogvlm_position_ids(context):
    """get cogvlm position_ids."""
    inputs = context.inputs
    q_seq_length = inputs.seq_length

    # avoid duplicated seq_len tolist
    context.q_seq_length_list = q_seq_length.tolist()
    vision_input_info = inputs.vision_inputs
    position_id_offsets = vision_input_info.history_image_token_lengths - vision_input_info.history_image_nums * 3
    if inputs.is_decoding:
        position_ids = inputs.history_lengths - position_id_offsets
    else:
        if vision_input_info.input_embeddings is not None and len(
                vision_input_info.input_embeddings) > 0:
            starts = inputs.history_lengths - vision_input_info.history_lengths
            ends = starts + q_seq_length
            token_type_ids = vision_input_info.input_embedding_indexing.to(
                 torch.int)
            
            # add token_type_range data_struct to context
            context.token_type_range, _ = get_range_ones(vision_input_info.input_embedding_indexing)
            history_position_lengths = vision_input_info.history_lengths - position_id_offsets
            position_ids_all = history_position_lengths[:,
                                                        None] + build_position_ids(
                                                            token_type_ids)
            position_ids = torch.cat([
                pids[s:e]
                for (pids, s, e) in zip(position_ids_all, starts, ends)
            ])
            vision_token_mask_all, _ = get_vision_expert_mask(token_type_ids)
            vision_token_mask = torch.cat([
                masks[s:e]
                for (masks, s, e) in zip(vision_token_mask_all, starts, ends)
            ])
            mask_indexing = torch.arange(vision_token_mask.shape[-1],
                                         device=vision_token_mask.device)
            vision_token_mask_new = mask_indexing[vision_token_mask]
            language_token_mask_new = mask_indexing[~vision_token_mask]

            context.vision_token_mask = vision_token_mask_new
            context.language_token_mask = language_token_mask_new

            # add vision & lang token range to context
            # vis_tok_mask_all is the original mask
            context.vision_token_range, context.language_token_range = get_range_ones(
                vision_token_mask_all)

        else:
            position_ids = context.attention_mask.long().cumsum(-1) - 1
            position_ids += (inputs.history_lengths -
                             position_id_offsets).unsqueeze(-1)
            device = position_ids.device
            position_ids_1d = [
                ids[:l]
                for ids, l in zip(position_ids.cpu(), q_seq_length.cpu())
            ]
            position_ids = torch.cat(position_ids_1d).to(device)

    return position_ids
