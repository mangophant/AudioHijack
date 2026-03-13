import os
import torch
import whisper
from transformers import AutoTokenizer, AutoConfig
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.llama_omni_src import OmniSpeechLlamaForCausalLM
from model.llama_omni_src import Omni2SpeechQwen2ForCausalLM


class LlamaOmniSeries(LALM):
    
    def __init__(self, config):
        super().__init__(config)
    
    def load(self, device, dtype):
        pass
    
    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors="pt", **kwargs)
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def expand_prompt(self, prompts, audio_lengths):
        audio_token = '<speech>'
        expanded_prompts = []
        for prompt in prompts:
            replace_str = []
            while audio_token in prompt:
                audio_length = audio_lengths.pop(0)
                num_audio_tokens = (audio_length + 1) // self.audio_token_stride
                expanded_audio_token = audio_token * num_audio_tokens
                replace_str.append(expanded_audio_token)
                prompt = prompt.replace(audio_token, "<placeholder>", 1)
            while "<placeholder>" in prompt:
                prompt = prompt.replace("<placeholder>", replace_str.pop(0), 1)
            expanded_prompts.append(prompt)
        return expanded_prompts


@LALMFactory.register('llama_omni')
class LlamaOmni(LlamaOmniSeries):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language.'
        self.system_prompt_template = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>'
        self.audio_prompt = '<speech>'
        self.user_prompt_template = '<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>'
        self.assistant_prompt_template = '<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>'
        self.generation_prefix = '<|start_header_id|>assistant<|end_header_id|>\n'
    
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path.lalm_path, use_fast=False)
        self.tokenizer.add_tokens(['<speech>'])
        self.tokenizer.padding_side = 'left'
        self.llm_base = OmniSpeechLlamaForCausalLM.from_pretrained(
            self.weight_path.lalm_path,
            low_cpu_mem_usage=False,
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        whisper_model = whisper.load_model(self.weight_path.encoder_path, device=device).to(dtype=dtype)
        self.llm_base.get_model().set_speech_encoder(whisper_model.encoder)
        self.llm_base.config.speech_token_id = self.tokenizer.convert_tokens_to_ids('<speech>')
        self.llm_base.config.pad_token_id = self.tokenizer.pad_token_id
        self.end_header_id = self.tokenizer.convert_tokens_to_ids('<|end_header_id|>')
        self.eot_id = self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        self.audio_token_id = self.llm_base.config.speech_token_id
        self.audio_token_stride = self.llm_base.get_speech_projector().k * 2
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
        
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        fbanks, audio_lengths = None, None
        if len(audio_inputs) > 0:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_token_stride]
            fbank_mask = fbank_mask.bool()
            audio_lengths = fbank_mask.sum(-1)
            text_inputs = self.expand_prompt(text_inputs, audio_lengths.tolist())
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'speech': fbanks,
            'speech_lengths': audio_lengths
        }
        input_length = 0
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            end_head_idx = (input_id == self.end_header_id).nonzero(as_tuple=True)[0].cpu().tolist()
            eot_idx = (input_id == self.eot_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(end_head_idx) > 0 and len(eot_idx) > 0:
                for b, e in zip(end_head_idx[1:-1], eot_idx[1:]):
                    input_mask[i, b+2:e] = 2 # text user prompt mask: 2
                input_mask[i, end_head_idx[0]+2:eot_idx[0]] = 3 # system / function prompt mask: 3
                input_mask[i, end_head_idx[-1]+2:] = 4 # target index mask: 4
            if len(audio_inputs) > 0:
                audio_idx = (input_id == self.audio_token_id).nonzero(as_tuple=True)[0]
                if audio_idx.numel() > 0:
                    input_mask[i, audio_idx] = fbank_token_mask[i, :len(audio_idx)]
        return input_params, input_length, input_mask
        

@LALMFactory.register('llama_omni2')
class LlamaOmni2(LlamaOmniSeries):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful assistant.'
        self.system_prompt_template = '<|im_start|>system\n\n{content}\n<|im_end|>'
        self.audio_prompt = '<speech>'
        self.user_prompt_template = '<|im_start|>user\n{content}\n<|im_end|>'
        self.assistant_prompt_template = '<|im_start|>assistant\n{content}\n<|im_end|>'
        self.generation_prefix = '<|im_start|>assistant\n'
    
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        config = AutoConfig.from_pretrained(self.weight_path.lalm_path)
        config.tts_tokenizer = os.path.join(self.weight_path.lalm_path, 'tts_tokenizer')
        config._attn_implementation = 'eager'
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path.lalm_path, use_fast=False)
        # self.tokenizer.add_tokens(['<speech>']) # Llama-Omini2 already adds the speech token
        self.tokenizer.padding_side = 'left'
        self.llm_base = Omni2SpeechQwen2ForCausalLM.from_pretrained(
            self.weight_path.lalm_path,
            config=config,
            device_map=device,
            torch_dtype=dtype)
        whisper_model = whisper.load_model(self.weight_path.encoder_path, device=device).to(dtype=dtype)
        self.llm_base.get_model().set_speech_encoder(whisper_model.encoder)
        self.llm_base.config.speech_token_id = self.tokenizer.convert_tokens_to_ids('<speech>')
        self.llm_base.config.pad_token_id = self.tokenizer.pad_token_id
        self.im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.audio_token_id = self.llm_base.config.speech_token_id
        self.audio_token_stride = self.llm_base.get_speech_projector().k * 2
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
        
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        fbanks, audio_lengths = None, None
        if len(audio_inputs) > 0:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_token_stride]
            fbank_mask = fbank_mask.bool()
            audio_lengths = fbank_mask.sum(-1)
            text_inputs = self.expand_prompt(text_inputs, audio_lengths.tolist())
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'speech': fbanks,
            'speech_lengths': audio_lengths
        }
        input_length = 0
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            im_start_idx = (input_id == self.im_start_id).nonzero(as_tuple=True)[0].cpu().tolist()
            im_end_idx = (input_id == self.im_end_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(im_end_idx) > 0 and len(im_start_idx) > 0:
                for b, e in zip(im_start_idx[1:-1], im_end_idx[1:]):
                    input_mask[i, b+3:e-1] = 2 # text user prompt mask: 2
                input_mask[i, im_start_idx[0]+3:im_end_idx[0]-1] = 3 # system / function prompt mask: 3
                input_mask[i, im_start_idx[-1]+3:] = 4 # target index mask: 4
            if len(audio_inputs) > 0:
                audio_idx = (input_id == self.audio_token_id).nonzero(as_tuple=True)[0]
                if audio_idx.numel() > 0:
                    input_mask[i, audio_idx] = fbank_token_mask[i, :len(audio_idx)]
        return input_params, input_length, input_mask