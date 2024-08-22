# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch.layers import (ApplyRotaryEmb, Attention, EmbeddingType,
                                     RMSNorm, SiluAndMul,
                                     build_rotary_embedding)
from lmdeploy.pytorch.layers.linear import (build_merged_colwise_linear,
                                            build_rowwise_linear)
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager

from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)

import torch._dynamo as dynamo
import torch_npu

class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        is_tp = world_size > 1
        self.ctx_mgr = ctx_mgr
        self.num_heads = origin.num_heads // world_size
        self.num_kv_heads = origin.num_key_value_heads // world_size
        self.head_dim = origin.head_dim

        # qkv
        self.qkv_proj = build_merged_colwise_linear(
            origin.q_proj,
            origin.k_proj,
            origin.v_proj,
            ctx_mgr=ctx_mgr,
            is_tp=is_tp,
        )
        del origin.q_proj, origin.k_proj, origin.v_proj
        self.qkv_weight = self.qkv_proj.get_parameter('impl.mod.weight').data
        
        dynamo.mark_static(self.qkv_weight)

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            v_head_size=self.head_dim,
        )

        self.o_proj = build_rowwise_linear(
            origin.o_proj,
            ctx_mgr=ctx_mgr,
            is_tp=is_tp,
        )
        self.o_weight = self.o_proj.get_parameter('impl.mod.weight').data
        dynamo.mark_static(self.o_weight)

    @staticmethod
    def _load_weights(mod, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear(getattr(mod, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(mod.o_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='o_proj')

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
        # qkv_states = self.qkv_proj(hidden_states)
        qkv_states = torch.ops.atb.linear(hidden_states, self.qkv_weight, None, False, True)
        ## (-1, heads, head_dim)
        # qkv_states = qkv_states.flatten(0, -2)
        # qkv_states = qkv_states.unflatten(-1, (-1, self.head_dim))
        qkv_states = qkv_states.view(-1, self.num_heads + self.num_kv_heads + self.num_kv_heads, self.head_dim)
        query_states, key_states, value_states = qkv_states.split(
            (
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
            ),
            dim=1,
        )

        cos, sin = rotary_pos_emb
        # import pdb;pdb.set_trace()
        # seqlen = query_states.shape[0]
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states.view(-1, self.num_heads * self.head_dim),
            key_states.view(-1, self.num_kv_heads * self.head_dim),
            cos,
            sin,
            attn_metadata.kv_seqlens_int,
            inplace=True,
        )
        
        # print('###')
        # import pdb;pdb.set_trace()
        # pass
        
        attn_output = self.attn_fwd(
            query_states.view(-1, self.num_heads, self.head_dim),
            key_states.view(-1, self.num_kv_heads, self.head_dim),
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            inplace=True,
        )
    
        # print('###')
        # import pdb;pdb.set_trace()
        # pass
    


        # attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        attn_output = attn_output.view(1, -1, self.num_heads * self.head_dim)


        # attn_output = self.o_proj(attn_output)
        attn_output = torch.ops.atb.linear(attn_output, self.o_weight, None, False, True)

        return attn_output


class LlamaMLP(nn.Module):

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        is_tp = world_size > 1
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            origin.gate_proj,
            origin.up_proj,
            ctx_mgr=ctx_mgr,
            is_tp=is_tp,
        )
        del origin.gate_proj, origin.up_proj

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(origin.down_proj,
                                              ctx_mgr=ctx_mgr,
                                              is_tp=is_tp)
        self.gate_up_weight = self.gate_up_proj.get_parameter('impl.mod.weight').data
        self.down_weight = self.down_proj.get_parameter('impl.mod.weight').data
        dynamo.mark_static(self.gate_up_weight)
        dynamo.mark_static(self.down_weight)
        
        
    @staticmethod
    def _load_weights(mod: nn.Module, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['gate_proj', 'up_proj']:
            colwise_parallelize_linear(getattr(mod, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(mod.down_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='down_proj')

    def forward(self, x):
        # # import pdb;pdb.set_trace()
        # gate_up = self.gate_up_proj(x)
        # act = self.act_fn(gate_up)
        # return self.down_proj(act)
        return torch.ops.atb.mlp_gate.default(x, self.gate_up_weight, self.down_weight)


class LlamaDecoderLayer(nn.Module):

    def __init__(self, origin: nn.Module, layer_idx: int,
                 ctx_mgr: StepContextManager):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(origin.self_attn, ctx_mgr)
        self.mlp = LlamaMLP(origin.mlp, ctx_mgr)

        # norm
        input_layernorm = origin.input_layernorm
        is_w8a8 = hasattr(input_layernorm, 'from_float')
        self.input_layernorm = RMSNorm(
            input_layernorm.weight,
            input_layernorm.variance_epsilon,
            is_w8a8=is_w8a8,
        )
        post_attention_layernorm = origin.post_attention_layernorm
        is_w8a8 = hasattr(post_attention_layernorm, 'from_float')
        self.post_attention_layernorm = RMSNorm(
            post_attention_layernorm.weight,
            post_attention_layernorm.variance_epsilon,
            is_w8a8=is_w8a8,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # print('###')
        # import pdb;pdb.set_trace()
        # pass

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class LlamaModel(nn.Module):

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        self.embed_tokens = origin.embed_tokens
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(layer, idx, ctx_mgr)
            for idx, layer in enumerate(origin.layers)
        ])
        norm = origin.norm
        is_w8a8 = hasattr(norm, 'from_float')
        self.norm = RMSNorm(norm.weight,
                            norm.variance_epsilon,
                            is_w8a8=is_w8a8)

        rotary_emb = origin.layers[0].self_attn.rotary_emb
        rotary_name = type(rotary_emb).__name__
        if rotary_name in [
                'LlamaRotaryEmbedding', 'LlamaLinearScalingRotaryEmbedding'
        ]:
            emb_type = EmbeddingType.LinearScaling
        elif rotary_name == 'LlamaDynamicNTKScalingRotaryEmbedding':
            emb_type = EmbeddingType.DynamicNTKScaling
        scaling_factor = getattr(rotary_emb, 'scaling_factor', 1.0)
        self.rotary_emb = build_rotary_embedding(
            rotary_emb.dim,
            rotary_emb.max_position_embeddings,
            rotary_emb.base,
            scaling_factor,
            emb_type,
        )

    def load_tensor(self, name):
        import pickle
        with open(f'{name}.pkl', 'rb') as f:
            out = pickle.load(f)
        return out.to('npu')
    
    @torch.compile(backend='atbgraph', dynamic=True)
    def call_layers(self, hidden_states, residual, past_key_values, rotary_pos_emb, attn_metadata):
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )
            break

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        
        
        # cos = self.load_tensor('cos')
        # sin = self.load_tensor('sin')
        # k_cache = self.load_tensor('k_cache')
        # v_cache = self.load_tensor('v_cache')
        # past_key_values[0] = (k_cache, v_cache)
        
        # rotary_pos_emb = (cos, sin)
        # # dynamo.mark_dynamic(attn_metadata.block_offsets_int, 0)
        # # dynamo.mark_dynamic(attn_metadata.block_offsets_int, 1)
        # hidden_states = self.load_tensor('hidden_states')
        # hidden_states = torch_npu.npu_format_cast(hidden_states, 2)

        # import pdb;pdb.set_trace()
        # pass
        hidden_states = self.call_layers(hidden_states, residual, past_key_values, rotary_pos_emb, attn_metadata)
        # hidden_states, residual = self.call_layers(hidden_states, residual, past_key_values, rotary_pos_emb, attn_metadata)
        # hidden_states, _ = self.norm(hidden_states, residual)
        # import pdb;pdb.set_trace()

        return hidden_states


class LlamaForCausalLM(nn.Module):

    support_cuda_graph = True

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        self.model = LlamaModel(origin.model, ctx_mgr)
        self.lm_head = build_rowwise_linear(origin.lm_head)
        self.opt_model = torch.compile(self.model, backend='atbgraph', dynamic=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states = self.model(
        # hidden_states = self.opt_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
