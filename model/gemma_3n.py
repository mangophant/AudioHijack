import math
import torch
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.gemma_3n_src import Gemma3nProcessor, Gemma3nForConditionalGeneration


@LALMFactory.register('gemma_3n')
class Gemma3n(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful assistant.'
        self.system_prompt_template = 'system\n{content}\n\n'
        self.audio_prompt = '\n\n<start_of_audio>' + '<audio_soft_token>' + '<end_of_audio>\n\n'
        self.user_prompt_template = '<start_of_turn>user\n{content}<end_of_turn>\n'
        self.assistant_prompt_template = '<start_of_turn>model\n{content}<end_of_turn>\n'
        self.generation_prefix = '<start_of_turn>model\n'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = Gemma3nProcessor.from_pretrained(self.weight_path.lalm_path).tokenizer
        self.tokenizer.padding_side = 'left'
        device_map = {
            "model.vision_tower": 0,
            "model.audio_tower": 0,
            "model.embed_audio": 0,
            "model.embed_vision": 0,
            "model.language_model": 1,
            "lm_head": 1,
        }
        self.llm_base = Gemma3nForConditionalGeneration.from_pretrained(
            self.weight_path.lalm_path,
            _attn_implementation='eager',
            device_map=device_map,
            max_memory={
                0: "46GiB",
                1: "46GiB",
            },
            torch_dtype=dtype)
        self.turn_start_id = self.tokenizer.convert_tokens_to_ids('<start_of_turn>')
        self.turn_end_id = self.tokenizer.convert_tokens_to_ids('<end_of_turn>')
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids('<start_of_audio>')
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids('<end_of_audio>')
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<audio_soft_token>')
        self.audio_length_per_tok = 16
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
    
    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def expand_prompt(self, prompts, audio_token_lengths):
        audio_token = '<audio_soft_token>'
        expanded_prompts = []
        for prompt in prompts:
            replace_str = []
            while audio_token in prompt:
                num_audio_tokens = audio_token_lengths.pop(0)
                expanded_audio_token = audio_token * num_audio_tokens
                replace_str.append(expanded_audio_token)
                prompt = prompt.replace(audio_token, "<placeholder>", 1)
            while "<placeholder>" in prompt:
                prompt = prompt.replace("<placeholder>", replace_str.pop(0), 1)
            expanded_prompts.append(prompt)
        return expanded_prompts

    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        fbanks, fbank_mask = None, None
        if len(audio_inputs) > 0:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(dtype=self.dtype)
            fbank_mask = fbank_mask.bool()
            audio_lengths = torch.ones(fbanks.shape[0]) * fbanks.shape[-1]
            audio_token_lengths = torch.ceil(audio_lengths / 3000 * 188).long().tolist()
            text_inputs = self.expand_prompt(text_inputs, audio_token_lengths)
            # fbank[1, D, T] -> input_features[1, T, D] -> input_features[B, T', D]
            fbanks = fbanks.transpose(1, 2).reshape(-1, 3000, self.fbank_config.n_mels)
            fbank_mask = fbank_mask.reshape(-1, 3000)
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids==self.audio_token_id] = 3
        input_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'input_features': fbanks,
            'input_features_mask': fbank_mask,
        }
        input_length = input_ids.size(1)
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            turn_start_idx = (input_id == self.turn_start_id).nonzero(as_tuple=True)[0].cpu().tolist()
            turn_end_idx = (input_id == self.turn_end_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(turn_end_idx) > 0 and len(turn_start_idx) > 0:
                for b, e in zip(turn_start_idx[:-1], turn_end_idx):
                    input_mask[i, b+3:e] = 2 # text user prompt mask: 2
                input_mask[i, 2:turn_start_idx[0]-1] = 3 # system / function prompt mask: 3
                input_mask[i, turn_start_idx[-1]+3:] = 4 # target index mask: 4
            if len(audio_inputs) > 0:
                audio_idx = (input_id == self.audio_token_id).nonzero(as_tuple=True)[0]
                audio_token_len = math.ceil(fbank_mask[i].sum() / fbank_mask.shape[-1] * 188)
                if audio_idx.numel() > 0:
                    input_mask[i, audio_idx[:audio_token_len]] = 1
                    input_mask[i, audio_idx[audio_token_len:]] = 0
        input_mask[(input_ids == self.audio_start_id) | (input_ids == self.audio_end_id)] = 0 # special token mask: 0
        return input_params, input_length, input_mask
    