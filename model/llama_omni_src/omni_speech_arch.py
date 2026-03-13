# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
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

import torch
from abc import ABC, abstractmethod
from whisper.model import LayerNorm
from .speech_projector import EncoderProjectorConcat


IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200


def replace_layer_norm(module):
    for name, child in module.named_children():
        if isinstance(child, LayerNorm):
            old_params = child.state_dict()
            device = next(child.parameters()).device
            dtype = next(child.parameters()).dtype
            new_layer_norm = torch.nn.LayerNorm(
                child.normalized_shape, 
                eps=child.eps, 
                elementwise_affine=child.elementwise_affine
            ).to(device=device, dtype=dtype)
            new_layer_norm.load_state_dict(old_params)
            new_layer_norm.to()
            setattr(module, name, new_layer_norm)
        else:
            replace_layer_norm(child)


class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)
        self.speech_encoder = None
        self.speech_projector = EncoderProjectorConcat(config)
    
    def set_speech_encoder(self, encoder):
        self.speech_encoder = encoder
        replace_layer_norm(self.speech_encoder)

    def get_speech_encoder(self):
        speech_encoder = getattr(self, "speech_encoder", None)
        return speech_encoder


class OmniSpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector

    def encode_speech(self, speech, speech_lengths):
        speech_encoder = self.get_speech_encoder()
        encoder_outs = speech_encoder(speech)
        speech_lengths = (speech_lengths + 1) // 2
        speech_projector = self.get_speech_projector()
        speech_features = speech_projector(encoder_outs)
        speech_lengths = speech_lengths // speech_projector.k
        return speech_features, speech_lengths

    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # extract special_speech_mask corresponding to '<speech>' first
        special_speech_mask = (input_ids == self.config.speech_token_id).to(input_ids.device)
        # encode speech to features and then extract them using mask
        speech_features, speech_output_lengths = self.encode_speech(speech, speech_lengths)
        max_speech_tokens = speech_features.shape[1]
        speech_features_mask = torch.arange(max_speech_tokens, device=speech_features.device)[None, :]
        speech_features_mask = speech_features_mask < speech_output_lengths[:, None]
        speech_features = speech_features[speech_features_mask]
        # make sure the numbers of speech token and feature match
        n_speech_tokens = special_speech_mask.sum().item()
        n_speech_features = speech_features.shape[0]
        if n_speech_tokens != n_speech_features:
            raise ValueError(
                f"speech features and speech tokens do not match: tokens: {n_speech_tokens}, features {n_speech_features}"
            )
        # token '<speech>' is not in the tokenizer vocab, we need to convert is to padding token before embedding
        input_ids = input_ids.masked_fill(special_speech_mask, self.config.pad_token_id)
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        special_speech_mask = special_speech_mask.unsqueeze(-1).expand_as(inputs_embeds)
        # fill inputs_embeds with speech features using special_speech_mask
        inputs_embeds = inputs_embeds.masked_scatter(special_speech_mask, speech_features)
        
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels