# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd, rms_norm
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)

from infer_ext.vendor.maca.maca_extension import ops as maca_ext_ops


class LlamaRMSNormMaca(nn.Module):
    """Rewrite RMSNorm."""

    def forward(self, hidden_states, residual: torch.Tensor = None):
        """forward."""
        # torch.nn.functional.normalize based implementation might leads
        # to wrong output
        ret = rms_norm(hidden_states, self.weight, self.variance_epsilon, residual)

        return ret


class PatchedInternLM2Attention(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['wqkv']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['wo']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of LlamaAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        position_ids_1d = context.position_ids_1d
        max_kv_seq_length = context.max_kv_seq_length

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            qkv_states = self.wqkv(hidden_states)
            qkv_states = rearrange(
                qkv_states,
                'b q (h gs d) -> (b q) h gs d',
                gs=2 + self.num_key_value_groups,
                d=self.head_dim,
            )
            query_states = qkv_states[..., :self.num_key_value_groups, :]
            query_states = query_states.flatten(1, 2)
            key_states = qkv_states[..., -2, :]
            value_states = qkv_states[..., -1, :]
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            # compat
            if not hasattr(self, '_use_old_rotary_emb'):
                import inspect
                args = inspect.getargspec(self.rotary_emb.forward)[0]
                self._use_old_rotary_emb = 'seq_len' in args

            if not hasattr(context, '_cos'):
                if self._use_old_rotary_emb:
                    kwargs = dict(seq_len=max_kv_seq_length)
                else:
                    kwargs = dict(position_ids=position_ids_1d[None])

                cos, sin = self.rotary_emb(value_states.transpose(0, 1),
                                           **kwargs)
                context._cos = cos
                context._sin = sin
            else:
                cos = context._cos
                sin = context._sin

            if self._use_old_rotary_emb:
                _position_ids_1d = position_ids_1d
            else:
                _position_ids_1d = torch.arange(0,
                                                len(position_ids_1d),
                                                device=query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids,
                position_ids_1d=_position_ids_1d,
                q_embed=query_states,
                k_embed=key_states)
            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

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
        )

        attn_output = query_states
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        attn_output = self.wo(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            world_size=world_size,
        )


class PatchedInternLM2AttentionAscend(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['wqkv']:
            colwise_parallelize_linear(
                getattr(self, mod_name),
                loader,
                rank=rank,
                world_size=world_size,
                prefix=mod_name,
            )
        for mod_name in ['wo']:
            rowwise_parallelize_linear(
                getattr(self, mod_name),
                loader,
                rank=rank,
                world_size=world_size,
                prefix=mod_name,
            )

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of LlamaAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        position_ids_1d = context.position_ids_1d
        max_kv_seq_length = context.max_kv_seq_length

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            qkv_states = self.wqkv(hidden_states)
            qkv_states = rearrange(
                qkv_states,
                'b q (h gs d) -> (b q) h gs d',
                gs=2 + self.num_key_value_groups,
                d=self.head_dim,
            )
            query_states = qkv_states[..., :self.num_key_value_groups, :]
            query_states = query_states.flatten(1, 2)
            key_states = qkv_states[..., -2, :].contiguous()
            value_states = qkv_states[..., -1, :]
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            # compat
            if not hasattr(self, '_use_old_rotary_emb'):
                import inspect

                args = inspect.getargspec(self.rotary_emb.forward)[0]
                self._use_old_rotary_emb = 'seq_len' in args

            if not hasattr(context, '_cos'):
                if self._use_old_rotary_emb:
                    kwargs = dict(seq_len=max_kv_seq_length)
                else:
                    kwargs = dict(position_ids=position_ids_1d[None])

                cos, sin = self.rotary_emb(value_states.transpose(0, 1),
                                           **kwargs)
                context._cos = cos
                context._sin = sin
            else:
                cos = context._cos
                sin = context._sin

            if self._use_old_rotary_emb:
                _position_ids_1d = position_ids_1d
            else:
                _position_ids_1d = torch.arange(0,
                                                len(position_ids_1d),
                                                device=query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids,
                position_ids_1d=_position_ids_1d,
                q_embed=query_states,
                k_embed=key_states,
                context=context,
            )
            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

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
            context=context,
        )

        attn_output = query_states
        paged_attention_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            context=context,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        attn_output = self.wo(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            world_size=world_size,
        )


class PatchedInternLM2AttentionMaca(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['wqkv']:
            colwise_parallelize_linear(
                getattr(self, mod_name),
                loader,
                rank=rank,
                world_size=world_size,
                prefix=mod_name,
            )
        for mod_name in ['wo']:
            rowwise_parallelize_linear(
                getattr(self, mod_name),
                loader,
                rank=rank,
                world_size=world_size,
                prefix=mod_name,
            )

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of LlamaAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        position_ids_1d = context.position_ids_1d
        max_kv_seq_length = context.max_kv_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_dim = self.head_dim

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            qkv_states = torch.matmul(hidden_states, self.wqkv)
            kv_dim_size = self.wqkv.shape[-1] // (self.num_key_value_groups + 2)
            q_dim_size = kv_dim_size * self.num_key_value_groups
            query_states, key_states, value_states = qkv_states.split([q_dim_size, kv_dim_size, kv_dim_size], dim=-1)
            return query_states, key_states, value_states.view(-1, num_kv_heads, self.head_dim)

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            # compat
            if not hasattr(self, '_use_old_rotary_emb'):
                import inspect

                args = inspect.getargspec(self.rotary_emb.forward)[0]
                self._use_old_rotary_emb = 'seq_len' in args

            if not hasattr(context, 'cos_sin_cache'):
                if self._use_old_rotary_emb:
                    kwargs = dict(seq_len=max_kv_seq_length)
                else:
                    kwargs = dict(position_ids=position_ids_1d[None])

                cos, sin = self.rotary_emb(value_states.transpose(0, 1),
                                           **kwargs)
                new_cos = cos.squeeze(-2)
                new_cos = new_cos[..., :new_cos.shape[-1] // 2]
                new_sin = sin.squeeze(-2)
                new_sin = new_sin[..., :new_sin.shape[-1] // 2]
                cos_sin_cache = torch.cat((new_cos, new_sin), dim=-1)
                context.cos_sin_cache = cos_sin_cache

            if self._use_old_rotary_emb:
                _position_ids_1d = position_ids_1d
            else:
                _position_ids_1d = torch.arange(0,
                                                len(position_ids_1d),
                                                device=query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                _position_ids_1d,
                context=context,
            )
            return query_states.view(-1, num_heads, self.head_dim), key_states.view(-1, num_kv_heads, self.head_dim), value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

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
            context=context,
        )

        attn_output = query_states
        paged_attention_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            max_kv_seq_length=max_kv_seq_length,
            window_size=None,
            context=context,
        )
        attn_output = attn_output.view(*hidden_states.shape[:-1], -1)

        attn_output = torch.matmul(attn_output, self.wo.weight)
        if self.wo.bias:
            attn_output = attn_output + self.wo.bias

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            world_size=world_size,
        )


class PatchedInternLM2MLP(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['w1', 'w3']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['w2']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedInternLM2MLPMaca(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['w1', 'w3']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['w2']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs

    def forward(self, x):
        # w1/w2/w3 bias is None for internlm2
        t = torch.matmul(x, self.trans_w13)
        t_shape = t.shape
        d = t_shape[-1] // 2
        output_shape = (t_shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        maca_ext_ops.silu_and_mul(out, t)
        down_proj = torch.matmul(out, self.w2.weight)

        return down_proj


class PatchedInternLM2Model(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # Attention mask is not necessary in continuous batching
        attention_mask = None
        hidden_states = inputs_embeds
        residual = None

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (past_key_values[idx]
                              if past_key_values is not None else None)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                residual = residual,
            )
            hidden_states, residual = layer_outputs[0], layer_outputs[1]


        hidden_states, _ = self.norm(hidden_states, residual)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_value,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
        )


# Modified from transformers.models.llama.modeling_llama.LlamaDecoderLayer with Llama->InternLM2
class PatchedInternLM2DecoderLayerMaca(nn.Module):
    """InternLM2 Decoder Layer.

    This module is a single layer of the InternLM2 model.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = PatchedInternLM2Attention(config=config, layer_idx=layer_idx)

        self.feed_forward = PatchedInternLM2MLP(config)
        self.attention_norm = LlamaRMSNormMaca(config.hidden_size,
                                               eps=config.rms_norm_eps)
        self.ffn_norm = LlamaRMSNormMaca(config.hidden_size,
                                         eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(hidden_states, residual)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        # hidden_states = residual + hidden_states

        # Fully Connected
        # residual = hidden_states
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        # import pdb; pdb.set_trace()
        hidden_states = self.feed_forward(hidden_states)
        # hidden_states = residual + hidden_states

        outputs = (hidden_states, residual)

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs
