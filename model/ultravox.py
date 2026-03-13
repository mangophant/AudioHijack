import json
import torch
from transformers import AutoTokenizer
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.ultravox_src import UltravoxConfig, UltravoxModel
from model.tools import parse_tool_call, call_tool


@LALMFactory.register('ultravox')
class Ultravox(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a friendly and helpful character. You love to answer questions for people.'
        self.system_prompt_template = '<|start_header_id|>system<|end_header_id|>{content}\n<|eot_id|>'
        self.system_tool_prompt_template = '{tools}{system_tool_prompt}'
        self.audio_prompt = '<|audio|>'
        self.user_prompt_template = '<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>'
        self.assistant_prompt_template = '<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>'
        self.tool_prompt_template = '<|start_header_id|>ipython<|end_header_id|>\n{content}<|eot_id|>'
        self.tool_call_template = '{content}'
        self.tool_call_limit = config.tool_call_limit
        self.tools, self.system_tool_prompt = None, ''
        self.generation_prefix = '<|start_header_id|>assistant<|end_header_id|>\n'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path.lalm_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config_dict, kwargs = UltravoxConfig.get_config_dict(self.weight_path.lalm_path)
        config_dict['text_model_id'] = self.weight_path.llm_path
        config_dict['audio_model_id'] = self.weight_path.encoder_path
        config = UltravoxConfig.from_dict(config_dict, **kwargs)
        self.llm_base = UltravoxModel.from_pretrained(
            self.weight_path.lalm_path,
            config=config,
            device_map=device,
            torch_dtype=dtype)
        self.audio_length_per_tok = 16
        self.audio_token = '<|audio|>'
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.start_header_id = self.tokenizer.convert_tokens_to_ids('<|start_header_id|>')
        self.end_header_id = self.tokenizer.convert_tokens_to_ids('<|end_header_id|>')
        self.eot_id = self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
    
    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def expand_prompt(self, prompts, audio_token_lengths):
        expanded_prompts = []
        for prompt in prompts:
            replace_str = []
            while self.audio_token in prompt:
                num_audio_tokens = audio_token_lengths.pop(0)
                expanded_audio_token = self.audio_token * num_audio_tokens
                replace_str.append(expanded_audio_token)
                prompt = prompt.replace(self.audio_token, "<placeholder>", 1)
            while "<placeholder>" in prompt:
                prompt = prompt.replace("<placeholder>", replace_str.pop(0), 1)
            expanded_prompts.append(prompt)
        return expanded_prompts

    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        has_audio_input = len(audio_inputs) > 0 and audio_inputs[0] is not None
        if has_audio_input:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(dtype=self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_length_per_tok]
            audio_lengths = fbank_mask.bool().sum(-1)
            audio_token_lengths = fbank_token_mask.bool().sum(-1)
            audio_batch_size = torch.ones(len(audio_inputs), 1).long().to(self.device)
            text_inputs = self.expand_prompt(text_inputs, audio_token_lengths.cpu().tolist())
            inputs = self.batch_encode(text_inputs)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            audio_token_mask = input_ids == self.audio_token_id
            audio_token_start_idx = (audio_token_mask).float().argmax(dim=1)
            input_ids[audio_token_mask] = self.tokenizer.pad_token_id
            input_params = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'audio_values': fbanks,
                'audio_lens': audio_lengths,
                'audio_batch_size': audio_batch_size,
                'audio_token_len': audio_token_lengths,
                'audio_token_start_idx': audio_token_start_idx
            }
        else:
            inputs = self.batch_encode(text_inputs)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            input_params = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        input_length = input_ids.size(1)
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            start_head_idx = (input_id == self.start_header_id).nonzero(as_tuple=True)[0].cpu().tolist()
            end_head_idx = (input_id == self.end_header_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(end_head_idx) > 0 and len(start_head_idx) > 0:
                for b, e in zip(end_head_idx[1:-1], start_head_idx[2:]):
                    input_mask[i, b+2:e] = 2 # text user prompt mask: 2
                input_mask[i, end_head_idx[0]+2:start_head_idx[1]] = 3 # system / function prompt mask: 3
                input_mask[i, end_head_idx[-1]+2:] = 4 # target index mask: 4
            input_mask[i][input_id == self.eot_id] = 0 # pad token id in each user/assistant prompt: 0
            if has_audio_input:
                audio_idx = audio_token_mask[i].nonzero(as_tuple=True)[0]
                if audio_idx.numel() > 0:
                    input_mask[i, audio_idx] = fbank_token_mask[i, :len(audio_idx)]
        return input_params, input_length, input_mask
    
    def query(self, messages):
        tool_calls = [[] for _ in range(len(messages))]
        responses = [[] for _ in range(len(messages))]
        for i in range(len(messages)):
            msgs = messages[i]
            _, resps = self.generate([msgs])
            responses[i].append(resps[0])
            tool = parse_tool_call(resps[0])
            call_count = self.tool_call_limit
            msgs = []
            while tool is not None and call_count > 0:
                name, args, syntax_success, execution_success, result = call_tool(tool)
                tool_calls[i].append((name, args, syntax_success, execution_success, result))
                content = resps[0].split('{')[0] + self.tool_call_template.format(content=json.dumps(tool))
                msgs.append({'role': 'assistant', 'content': content})
                msgs.append({'role': 'tool', 'content': result['result']})
                _, resps = self.generate([msgs])
                responses[i].append(resps[0])
                tool = parse_tool_call(resps[0])
                call_count -= 1
        return tool_calls, responses