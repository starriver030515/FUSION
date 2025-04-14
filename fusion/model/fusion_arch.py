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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import random

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_aux_vision_projector
from .vision_sampler import VisionTokenSampler
from fusion.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, LATENT_IMAGE_TOKEN_INDEX, INSTRUCT_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import math
from fusion.mm_utils import get_anyres_image_grid_shape

class FusionMetaModel:

    def __init__(self, config):
        super(FusionMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.aux_projector = build_aux_vision_projector(config)
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
            
            # our interaction layers
            num_of_vision_sampler_layers = config.num_of_vision_sampler_layers
            self.vision_sampler_layers = nn.ModuleList([VisionTokenSampler(config.hidden_size, config.hidden_size, config.hidden_size, 1) for _ in range(0, num_of_vision_sampler_layers)])

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        query_len = model_args.query_len
        window_size = model_args.window_size
        
        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.query_len = query_len
        self.config.window_size = window_size

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            self.aux_projector = build_aux_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

            num_of_vision_sampler_layers = self.config.num_of_vision_sampler_layers = model_args.num_of_vision_sampler_layers
            self.config.start_of_vision_sampler_layers = model_args.start_of_vision_sampler_layers
            self.config.stride_of_vision_sampler_layers = model_args.stride_of_vision_sampler_layers
            self.vision_sampler_layers = nn.ModuleList([VisionTokenSampler(self.config.hidden_size, self.config.hidden_size, self.config.hidden_size, 1) for _ in range(0, num_of_vision_sampler_layers)])
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
            for p in self.aux_projector.parameters():
                p.requires_grad = True
            for p in self.vision_sampler_layers.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            self.aux_projector.load_state_dict(get_w(mm_projector_weights, 'aux_projector'))

            self.vision_sampler_layers.load_state_dict(get_w(mm_projector_weights, 'vision_sampler_layers'))

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class FusionMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, instruct_embeddings, instruct_mask):
        # encoder image with instructs.
        images = images.flatten(0, 1)

        instruct_states = self.get_model().aux_projector(instruct_embeddings)

        image_features, image_forward_outs = self.get_model().get_vision_tower()(images, instruct_states, instruct_mask)
        image_features = self.get_model().get_vision_tower().feature_connector(image_features, image_forward_outs)
        image_features = self.get_model().mm_projector(image_features)

        height = width = self.get_model().get_vision_tower().num_patches_per_side
        vision_embeddings = image_forward_outs.hidden_states[0][:, :height*width]
        vision_features = self.get_model().aux_projector(image_features[:,:height*width])

        return image_features, vision_embeddings, vision_features
    
    def encode_global_images(self, global_images):
        # encoder global images without instructs.
        B = global_images.shape[0]
        global_images = global_images.flatten(0, 1)

        global_image_features, global_image_forward_outs = self.get_model().get_vision_tower()(global_images, None, None)
        global_image_features = self.get_model().get_vision_tower().feature_connector(global_image_features, global_image_forward_outs)
        global_image_features = self.get_model().mm_projector(global_image_features)

        height = width = self.get_model().get_vision_tower().num_patches_per_side
        global_image_features = global_image_features.contiguous().view(B, 4, -1, global_image_features.shape[-1])
        global_image_features = global_image_features[:,:, :height*width].contiguous().view(B, -1, global_image_features.shape[-1])
        global_image_features = global_image_features.view(B, 2, 2, height, width, -1)
        global_image_features = global_image_features.permute(0, 1, 3, 2, 4, 5).contiguous()
        global_image_features = global_image_features.flatten(1, 4)

        return global_image_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None, instructs=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None, None, None, None, None, None

        # we only calculate the instruct that is not padding(0)
        instruct_mask = instructs.ne(0)
        instruct_embeddings = self.get_model().embed_tokens(instructs)

        split_images, global_images = images[:,4:], images[:,:4]
        height = width = self.get_model().get_vision_tower().num_patches_per_side

        # we encode the split images with the instructs, which will be used as image token in input sequence.
        image_features, vision_embeddings, vision_features = self.encode_images(split_images, instruct_embeddings, instruct_mask)
        image_features, instruct_features = image_features[...,:height*width,:], image_features[...,height*width:,:]

        # If you want to use FUSION-L, you can uncomment the following lines.
        # image_features = image_features.view(image_features.shape[0], height, width, -1)
        # image_features = image_features.permute(0, 3, 1, 2)
        # image_features = nn.functional.interpolate(image_features, size=(12, 12), mode='bilinear', align_corners=False)
        # image_features = image_features.permute(0, 2, 3, 1).flatten(1, 2)

        # we encode the global images with no instructs, which will be used as query in decoder's interaction layer.
        global_image_features = self.encode_global_images(global_images)

        global_height = global_width = int(math.sqrt(global_image_features.shape[1]))
        query_list = self.config.query_len.split(',')
        query_len = int(random.choice(query_list))
        avg_height = avg_width = int(math.sqrt(query_len))

        # we init the context-aware latent token as pooled global image features.
        init_latent_features = global_image_features.view(global_image_features.shape[0], global_height, global_width, -1)
        init_latent_features = init_latent_features.permute(0, 3, 1, 2)
        init_latent_features = nn.functional.adaptive_max_pool2d(init_latent_features, (avg_height, avg_width))
        init_latent_features = init_latent_features.permute(0, 2, 3, 1).flatten(1, 2)

        # we pooled the global image features to the size: window_size*query_size.
        global_image_features = global_image_features.view(global_image_features.shape[0], global_height, global_width, -1)
        global_image_features = global_image_features.permute(0, 3, 1, 2)
        global_image_features = nn.functional.adaptive_max_pool2d(global_image_features, (avg_height*self.config.window_size, avg_width*self.config.window_size))
        global_image_features = global_image_features.permute(0, 2, 3, 1).flatten(1, 2)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            # special here! This corresponds to eval or inference scenario where labels are not provided. Here we set the last token to be the INSTRUCT_INDEX.
            labels[:, -1] = INSTRUCT_INDEX

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        vision_positions = []
        instruct_positions = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            vision_position = []
            instruct_position = []
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item() + (cur_input_ids == LATENT_IMAGE_TOKEN_INDEX).sum().item()
            image_token_indices = [-1] + torch.where((cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == LATENT_IMAGE_TOKEN_INDEX))[0].tolist() + [len(cur_input_ids)]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            total_len = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                total_len += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    has_image = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()
                    if not has_image:
                        # No image token in the input, this corresponds to pure language conversation
                        # To keep the format consistent, we set the image token embedding to 0 and add the corresponding mask
                        cur_image_features = torch.zeros_like(init_latent_features[cur_image_idx])
                        vision_position.extend(torch.arange(total_len, total_len + cur_image_features.shape[0], dtype=torch.long, device=cur_image_features.device))
                        total_len += cur_image_features.shape[0]
                    else:
                        # Image token in the input.
                        # According to our input format, for the first question, the image token is after latent image tokens. But here, we always put the image token in the first position.
                        if i == 0:
                            cur_image_features = image_features[cur_image_idx]
                            total_len += cur_image_features.shape[0]
                        else:
                            cur_image_features = init_latent_features[cur_image_idx]
                            vision_position.extend(torch.arange(total_len, total_len + cur_image_features.shape[0], dtype=torch.long, device=cur_image_features.device))
                            total_len += cur_image_features.shape[0]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                if i == num_images:
                    cur_image_idx += 1

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # Here, we get the final position of the latent image token and the last instruction token.
            # We then set the instruction token to be the IGNORE_INDEX as usual.
            instruct_position = torch.where(cur_new_labels == INSTRUCT_INDEX)[0].tolist()

            vision_positions.append(vision_position)
            instruct_positions.append(instruct_position)

            cur_new_labels[torch.where(cur_new_labels == INSTRUCT_INDEX)[0]] = IGNORE_INDEX
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # In the decoding process, we will use all the queries, so we must ensure that the query length is not truncated
            # We must ensure the integrity of the latent query and its question
            instruct_positions = [torch.tensor(ip, dtype=torch.long, device=self.device) for ip in instruct_positions]
            vision_positions = [torch.tensor(vp, dtype=torch.long, device=self.device) for vp in vision_positions]

            instruct_positions = [x[x < tokenizer_model_max_length] for x in instruct_positions]
            vision_positions = [
                x[:query_len * len(instruct_positions[i])] for i, x in enumerate(vision_positions)
            ]

            vision_positions = torch.nn.utils.rnn.pad_sequence(vision_positions, batch_first=True, padding_value=-1)
            instruct_positions = torch.nn.utils.rnn.pad_sequence(instruct_positions, batch_first=True, padding_value=-1)

            if instruct_positions.size(1) == 0:
                print("Warning: all instructions are truncated. This should not happen.")
                instruct_positions = torch.full((instruct_positions.size(0), 1), -1, dtype=torch.long, device=self.device)
                vision_positions = torch.full((vision_positions.size(0), query_len), -1, dtype=torch.long, device=self.device)
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # We pass instruct_embeddings, instruct_features, instruct_mask, vision_embeddings, vision_features parameters for calculating mapping loss
        # We pass global_image_features, query_len, vision_positions, instruct_positions parameters for interaction layers
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, instruct_embeddings, instruct_features, instruct_mask, vision_embeddings, vision_features, global_image_features, query_len, instruct_positions, vision_positions

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
