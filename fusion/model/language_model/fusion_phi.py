# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git
#
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Model, Phi3Config, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from ..fusion_arch import FusionMetaModel, FusionMetaForCausalLM


class FusionPhiConfig(Phi3Config):
    model_type = "fusion_phi"


class FusionPhiModel(FusionMetaModel, Phi3Model):
    config_class = FusionPhiConfig

    def __init__(self, config: Phi3Config):
        super(FusionPhiModel, self).__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        global_ctx: Optional[torch.Tensor] = None,
        vision_pos: Optional[torch.Tensor] = None,
        instruct_pos: Optional[torch.Tensor] = None,
        query_len: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if hidden_states.size(1) != 1:
                cross_layers_start_idx = self.config.start_of_vision_sampler_layers
                cross_index_step = self.config.stride_of_vision_sampler_layers
                cross_layers_start_idx_list = [cross_layers_start_idx+cross_index*cross_index_step for cross_index in range(len(self.vision_sampler_layers))]
                if i in cross_layers_start_idx_list:
                    B, S, D = hidden_states.size()
                    L = instruct_pos.size(1)
                    V = L * query_len
                    assert V == vision_pos.size(1)

                    # support beam search
                    # instruct_pos = instruct_pos.expand(B, -1)
                    # vision_pos = vision_pos.expand(B, -1)
                    # global_ctx = global_ctx.expand(B, -1, -1)

                    instruct_mask = instruct_pos != -1  # (B, L)
                    safe_instruct_pos = instruct_pos.masked_fill(~instruct_mask, 0)  # (B, L)
                    selected_instruct = hidden_states.gather(1, safe_instruct_pos.unsqueeze(-1).expand(-1, -1, D))  # (B, L, D)
                    mask_instruct = selected_instruct * (instruct_mask.unsqueeze(-1))
                    queries = mask_instruct.unsqueeze(2).expand(-1, -1, query_len, -1).contiguous().view(B, V, D)  # (B, V, D)

                    vision_mask = vision_pos != -1  # (B, V)
                    safe_vision_pos = vision_pos.masked_fill(~vision_mask, 0)  # (B, V)
                    selected_vision = hidden_states.gather(1, safe_vision_pos.unsqueeze(-1).expand(-1, -1, D))  # (B, V, D)
                    image_tokens = selected_vision * vision_mask.unsqueeze(-1) 

                    if self.gradient_checkpointing and self.training:
                        queries = self._gradient_checkpointing_func(
                            self.vision_sampler_layers[cross_layers_start_idx_list.index(i)].__call__,
                            hidden_states,
                            global_ctx,
                            vision_pos,
                            instruct_pos,
                            queries,
                            image_tokens,
                            query_len,
                            use_reentrant=False,
                        )
                    else:
                        queries = self.vision_sampler_layers[cross_layers_start_idx_list.index(i)](hidden_states, global_ctx, vision_pos, instruct_pos, queries, image_tokens, query_len)

                    B_idx, V_idx = torch.nonzero(vision_mask, as_tuple=True)  # Indices where vision_pos is not -1
                    positions = safe_vision_pos[B_idx, V_idx]  # (num_valid,)
                    out_valid = queries[B_idx, V_idx, :]  # (num_valid, q_dim)

                    positions = B_idx * S + positions
                    positions = positions.unsqueeze(-1).expand(-1, D)

                    new_hidden_states = hidden_states.flatten(0, 1)
                    new_hidden_states = new_hidden_states.scatter(0, positions, out_valid)
                    hidden_states = new_hidden_states.contiguous().view(B, S, D)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class FusionPhiForCausalLM(Phi3ForCausalLM, FusionMetaForCausalLM):
    config_class = FusionPhiConfig

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = FusionPhiModel(config)
        self.pretraining_tp = getattr(config, 'pretraining_tp', 1)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        instructs: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(f'input_ids: {input_ids}')
        instruct_embeddings, instruct_features, instruct_mask = None, None, None
        vision_embeddings, vision_features = None, None
        instruct_pos, vision_pos = None, None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                instruct_embeddings,
                instruct_features,
                instruct_mask,
                vision_embeddings,
                vision_features,
                global_context_feature,
                query_len,
                instruct_pos,
                vision_pos,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                instructs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            global_ctx=self.global_ctx if hasattr(self, 'global_ctx') else global_context_feature,
            vision_pos=self.vision_pos if hasattr(self, 'vision_pos') else vision_pos,
            instruct_pos=self.instruct_pos if hasattr(self, 'instruct_pos') else instruct_pos,
            query_len=self.query_len if hasattr(self, 'query_len') else query_len,
        )

        hidden_states = outputs[0]
        if getattr(self.config, 'pretraining_tp', 1) > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if instruct_embeddings is not None:
            batch_size, seq_length, hidden_size = instruct_embeddings.size()
            instruct_embeddings = instruct_embeddings.contiguous().view(batch_size * seq_length, hidden_size)
            instruct_features = instruct_features.contiguous().view(batch_size * seq_length, hidden_size)
            instruct_mask = instruct_mask.contiguous().view(batch_size * seq_length)
            instruct_embeddings = instruct_embeddings[instruct_mask == 1]
            instruct_features = instruct_features[instruct_mask == 1]

            batch_size, seq_length, hidden_size = vision_embeddings.size()
            vision_embeddings = vision_embeddings.contiguous().view(batch_size * seq_length, hidden_size)
            vision_features = vision_features.contiguous().view(batch_size * seq_length, hidden_size)
                
            instruct_embeddings = F.normalize(instruct_embeddings, p=2, dim=-1)
            instruct_features = F.normalize(instruct_features, p=2, dim=-1)
            vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
            vision_features = F.normalize(vision_features, p=2, dim=-1)

            additional_loss = 0.0
            if instruct_embeddings.size(0) > 0:
                target = torch.ones(instruct_embeddings.size(0), device=instruct_embeddings.device)
                instruct_similarity_loss = CosineEmbeddingLoss()(instruct_embeddings, instruct_features, target)
                additional_loss += instruct_similarity_loss

            if vision_embeddings.size(0) > 0:
                target = torch.ones(vision_embeddings.size(0), device=vision_embeddings.device)
                vision_similarity_loss = CosineEmbeddingLoss()(vision_embeddings, vision_features, target)
                additional_loss += vision_similarity_loss


            lambda_similarity = 0.1

            loss = loss + lambda_similarity * additional_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        instructs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        global_context_feature = None
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _,
                _,
                _,
                _,
                global_context_feature,
                query_len,
                instruct_pos,
                vision_pos,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                labels,
                images,
                image_sizes=image_sizes,
                instructs=instructs,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        self.global_ctx = global_context_feature
        self.query_len = query_len
        self.instruct_pos = instruct_pos
        self.vision_pos = vision_pos

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,      
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        instructs = kwargs.pop("instructs", None)
        labels = kwargs.pop("labels", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if instructs is not None:
            inputs['instructs'] = instructs
        if labels is not None:
            inputs['labels'] = labels
        return inputs


AutoConfig.register("fusion_phi", FusionPhiConfig)
AutoModelForCausalLM.register(FusionPhiConfig, FusionPhiForCausalLM)