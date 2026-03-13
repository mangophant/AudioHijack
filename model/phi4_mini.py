import json
import torch
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogramE
from model.phi4_mini_src import Phi4MMProcessor, Phi4MMForCausalLM
from model.tools import parse_tool_call, call_tool


@LALMFactory.register('phi4_mini')
class Phi4Mini(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful assistant.'
        self.system_prompt_template = '<|system|>{content}<|end|>'
        self.system_tool_prompt_template = '<|tool|>{tools}<|/tool|>{system_tool_prompt}'
        self.audio_prompt = '<|audio_1|>'
        self.user_prompt_template = '<|user|>{content}<|end|>'
        self.assistant_prompt_template = '<|assistant|>{content}<|end|>'
        self.tool_prompt_template = '<|tool_response|>{content}<|end|>'
        self.tool_call_template = '<|tool_call|>[\n{content}\n]<|/tool_call|>'
        self.tool_call_limit = config.tool_call_limit
        self.tools, self.system_tool_prompt = None, ''
        self.generation_prefix = '<|assistant|>'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogramE(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = Phi4MMProcessor.from_pretrained(self.weight_path.lalm_path).tokenizer
        self.tokenizer.padding_side = 'left'
        self.llm_base = Phi4MMForCausalLM.from_pretrained(
            self.weight_path.lalm_path,
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        self.invstd = self.llm_base.model.embed_tokens_extend.audio_embed.encoder.encoder_embedding.global_invstd
        self.audio_length_per_tok = 8
        self.system_id = self.tokenizer.convert_tokens_to_ids('<|system|>')
        self.user_id = self.tokenizer.convert_tokens_to_ids('<|user|>')
        self.assist_id = self.tokenizer.convert_tokens_to_ids('<|assistant|>')
        self.end_id = self.tokenizer.convert_tokens_to_ids('<|end|>')
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext11|>')
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()

    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def expand_prompt(self, prompts, audio_token_lengths):
        audio_token = '<|audio_1|>'
        expanded_prompts = []
        for prompt in prompts:
            replace_str = []
            while audio_token in prompt:
                num_audio_tokens = audio_token_lengths.pop(0)
                expanded_audio_token = '<|endoftext11|>' * num_audio_tokens
                replace_str.append(expanded_audio_token)
                prompt = prompt.replace(audio_token, "<placeholder>", 1)
            while "<placeholder>" in prompt:
                prompt = prompt.replace("<placeholder>", replace_str.pop(0), 1)
            expanded_prompts.append(prompt)
        return expanded_prompts

    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        fbanks, fbank_mask, audio_token_lengths = None, None, None
        has_audio_input = len(audio_inputs) > 0 and audio_inputs[0] is not None
        if has_audio_input:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.transpose(1, 2).to(dtype=self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_length_per_tok]
            audio_token_lengths = fbank_token_mask.bool().sum(-1)
            text_inputs = self.expand_prompt(text_inputs, audio_token_lengths.cpu().tolist())
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'input_mode': 2, # speech mode
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_audio_embeds": fbanks,
            "audio_embed_sizes": audio_token_lengths,
            "audio_attention_mask": fbank_mask
        }
        input_length = input_ids.size(1)
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            end_idx = (input_id == self.end_id).nonzero(as_tuple=True)[0].cpu().tolist()
            system_idx = (input_id == self.system_id).nonzero(as_tuple=True)[0].cpu().tolist()
            user_idx = (input_id == self.user_id).nonzero(as_tuple=True)[0].cpu().tolist()
            assist_idx = (input_id == self.assist_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(system_idx) > 0:
                system_end_idx = end_idx.pop(0)
                input_mask[i, system_idx[0]+1:system_end_idx] = 3 # system / function prompt mask: 3
            if len(assist_idx) > 0:
                for context_start_idx in sorted(user_idx + assist_idx[:-1]):
                    context_end_idx = end_idx.pop(0)
                    if context_start_idx < context_end_idx:
                        input_mask[i, context_start_idx+1:context_end_idx] = 2 # text user prompt mask: 2
                input_mask[i, assist_idx[-1]+1:] = 4 # target index mask: 4
            audio_idx = (input_id == self.audio_token_id).nonzero(as_tuple=True)[0]
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