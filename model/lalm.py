import os
import json
import base64
from pathlib import Path
import torch
from transformers import GenerationConfig

import util


class LALM(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.name = config.name
        self.scheme = config.scheme
        self.audio_type = config.audio_type
        self.voice_chat = config.voice_chat
        self.tool_use = config.tool_use
        self.accum_grad = config.accum_grad
        self.weight_path = config.weight_path
        self.parameter = config.parameter
        self.generate_kwargs = GenerationConfig(**config.generate_kwargs)
        if hasattr(config, 'fbank_config'):
            self.fbank_config = config.fbank_config
            self.sample_rate = self.fbank_config.sample_rate
            self.max_audio_len = self.fbank_config.n_samples
        
    def load(self, device, dtype):
        pass
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def batch_encode(self, texts, **kwargs):
        pass
    
    def batch_decode(self, ids):
        pass
    
    def apply_template(self, messages, add_prefix=True):
        text_inputs, audio_inputs = [], []
        for message in messages:
            audio = None
            full_system_prompt = self.system_prompt
            if self.tool_use and self.tools is not None:
                full_system_prompt += self.system_tool_prompt_template.format(
                    tools=json.dumps(self.tools, indent=2),
                    system_tool_prompt=self.system_tool_prompt
                )
            prompt = self.system_prompt_template.format(content=full_system_prompt)
            for msg in message:
                if msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        prompt += self.user_prompt_template.format(content=msg['content'])
                    elif isinstance(msg['content'], list):
                        content = ''
                        for item in msg['content']:
                            if item['type'] == 'text':
                                content += item['text']
                            elif item['type'] == 'input_audio':
                                content += self.audio_prompt
                                audio = item['input_audio']
                        prompt += self.user_prompt_template.format(content=content)
                elif msg['role'] == 'assistant':
                    prompt += self.assistant_prompt_template.format(content=msg['content'])
                elif msg['role'] == 'tool':
                    prompt += self.tool_prompt_template.format(content=msg['content'])
            if add_prefix:
                prompt += self.generation_prefix
            text_inputs.append(prompt)
            audio_inputs.append(audio)
        return text_inputs, audio_inputs
    
    @classmethod
    def create_prompt(
        cls,
        user_prompt,
        audio_data,
        encode_audio=False,
        max_audio_len=480000,
        sample_rate=16000
    ):
        is_voice_chat = os.path.exists(user_prompt) and user_prompt.endswith('.wav')
        if is_voice_chat:
            speech_prompt = util.load_audio(user_prompt, sr=sample_rate).to(audio_data.device)
            speech_prompt = speech_prompt[:max_audio_len - audio_data.shape[-1]]
            if encode_audio:
                audio = torch.cat((audio_data, speech_prompt), dim=-1)
                audio_bytes = util.b64encode_audio(audio)
                input_audio = {'data': audio_bytes, 'format': 'wav'}
            else:
                input_audio = {'audio_data': audio_data, 'speech_prompt': speech_prompt}
            return {
                'role': 'user', 
                'content': [{'type': 'input_audio', 'input_audio': input_audio}]
            }
        else:
            if encode_audio:
                audio_bytes = util.b64encode_audio(audio_data)
                input_audio = {"data": audio_bytes, "format": "wav"}
            else:
                input_audio = audio_data
            content = [
                {'type': 'input_audio', 'input_audio': input_audio},
                {'type': 'text', 'text': user_prompt}
            ]
            return {'role': 'user', 'content': content}
    
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        pass
    
    def pack_output(self, ids, mask):
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        return ids, mask
    
    def prepare_output(self, labels):
        targets = self.batch_encode(labels, add_special_tokens=False)
        target_ids = targets['input_ids']
        target_mask = targets['attention_mask'].bool()
        target_ids, target_mask = self.pack_output(target_ids, target_mask)
        return target_ids, target_mask
    
    def forward(self, messages, labels):
        text_inputs, audio_inputs = self.apply_template(messages, add_prefix=True)
        input_params, _, input_mask = self.pack_input(text_inputs, audio_inputs, labels, 'forward')
        target_ids, target_mask = self.prepare_output(labels)
        target_ids[~target_mask] = self.parameter.ignore_index
        output = self.llm_base(**input_params, output_attentions=True, return_dict=True)
        logits = output.logits[:, -target_ids.shape[-1]-1:-1, :]
        logits = logits.contiguous().view(-1, logits.size(-1)).float()
        targets = target_ids.contiguous().view(-1)
        attentions = output.attentions
        return logits, targets, attentions, input_mask
    
    def generate(self, messages):
        text_inputs, audio_inputs = self.apply_template(messages, add_prefix=True)
        input_params, input_length, _ = self.pack_input(text_inputs, audio_inputs, None, 'generate')
        output = self.llm_base.generate(
            **input_params, generation_config=self.generate_kwargs,
            return_dict_in_generate=True, output_logits=True
        )
        logits = torch.vstack(output.logits).float()
        output_ids = output.sequences[:, input_length:]
        responses = self.batch_decode(output_ids)
        return logits, responses
    
    
class LALMFactory:
    
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def create(cls, name, config):
        if name not in cls._registry:
            raise ValueError(f"Unknown LALM model: {name}")
        return cls._registry[name](config)