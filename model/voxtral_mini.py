import re
import json
import torch
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.voxtral_mini_src import MistralCommonTokenizer, VoxtralForConditionalGeneration
from model.tools import parse_tool_call, call_tool


@LALMFactory.register('voxtral_mini')
class VoxtralMini(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful assistant.'
        self.system_prompt_template = '<s>[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]'
        self.system_tool_prompt_template = '[AVAILABLE_TOOLS]{tools}[/AVAILABLE_TOOLS]{system_tool_prompt}'
        self.audio_prompt = '[BEGIN_AUDIO][AUDIO]'
        self.user_prompt_template = '[INST]{content}[/INST]'
        self.assistant_prompt_template = '{content}</s>'
        self.tool_prompt_template = '[TOOL_RESULTS]{content}[/TOOL_RESULTS]'
        self.tool_call_template = '[TOOL_CALLS][\n{content}\n]'
        self.tool_call_limit = config.tool_call_limit
        self.tools, self.system_tool_prompt = None, ''
        self.generation_prefix = ''
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = MistralCommonTokenizer.from_pretrained(self.weight_path.lalm_path)
        self.tokenizer.padding_side = 'left'
        self.llm_base = VoxtralForConditionalGeneration.from_pretrained(
            self.weight_path.lalm_path,
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        self.special_token_dict = {
            '<unk>': 0, '<s>': 1, '</s>': 2, '[INST]': 3, '[/INST]': 4, 
            '[AVAILABLE_TOOLS]': 5, '[/AVAILABLE_TOOLS]': 6, '[TOOL_RESULTS]': 7, '[/TOOL_RESULTS]': 8, '[TOOL_CALLS]': 9, 
            '[IMG]': 10, '<pad>': 11, '[IMG_BREAK]': 12, '[IMG_END]': 13, '[PREFIX]': 14, '[MIDDLE]': 15, '[SUFFIX]': 16, 
            '[SYSTEM_PROMPT]': 17, '[/SYSTEM_PROMPT]': 18, '[TOOL_CONTENT]': 19, 
            '[AUDIO]': 24, '[BEGIN_AUDIO]': 25, '[TRANSCRIBE]': 34}
        self.special_token_pattern = "(" + "|".join(re.escape(tok) for tok in sorted(self.special_token_dict.keys(), key=len, reverse=True)) + ")"
        self.audio_length_per_tok = self.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.audio_length_per_tok
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()

    def batch_encode(self, texts, **kwargs):
        batch_ids = []
        pad_token_id = self.special_token_dict['<pad>']
        for text in texts:
            ids = []
            for part in re.split(self.special_token_pattern, text):
                if part in self.special_token_dict.keys():
                    ids.append(self.special_token_dict[part])
                else:
                    ids += self.tokenizer.encode(part, add_special_tokens=False)
            batch_ids.append(ids)
        max_len = max(len(ids) for ids in batch_ids)
        input_ids, attention_mask = [], []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids = [pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            input_ids.append(padded_ids)
            attention_mask.append(mask)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def expand_prompt(self, prompts, audio_token_lengths):
        audio_token = '[AUDIO]'
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
        fbanks, fbank_token_mask = None, None
        has_audio_input = len(audio_inputs) > 0 and audio_inputs[0] is not None
        if has_audio_input:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(dtype=self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_length_per_tok]
            audio_lengths = torch.ones(fbanks.shape[0]) * fbanks.shape[-1]
            audio_token_lengths = torch.ceil(audio_lengths / self.audio_length_per_tok).long().tolist()
            text_inputs = self.expand_prompt(text_inputs, audio_token_lengths)
            fbanks = fbanks.reshape(self.fbank_config.n_mels, -1, 3000).transpose(0, 1)
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_features': fbanks,
            'audio_embed_mask': fbank_token_mask.bool() if fbank_token_mask is not None else None
        }
        input_length = input_ids.size(1)
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            binst_idx = (input_id == self.special_token_dict['[INST]']).nonzero(as_tuple=True)[0].cpu().tolist()
            einst_idx = (input_id == self.special_token_dict['[/INST]']).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(binst_idx) > 0 and len(einst_idx) > 0:
                for b, e in zip(binst_idx, einst_idx):
                    input_mask[i, b+1:e] = 2        # text user prompt mask: 2
                input_mask[i, :binst_idx[0]] = 3    # system / function prompt mask: 3
                input_mask[i, einst_idx[-1]+1:] = 4 # target index mask: 4
            audio_idx = (input_id == self.special_token_dict['[AUDIO]']).nonzero(as_tuple=True)[0]
            if audio_idx.numel() > 0:
                input_mask[i, audio_idx] = fbank_token_mask[i, :len(audio_idx)]
        input_mask[(input_ids < 20) | (input_ids == 25)] = 0 # special token mask: 0
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