import torch
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.qwen2_audio_src import Qwen2AudioProcessor, Qwen2AudioForConditionalGeneration


@LALMFactory.register('qwen2_audio')
class Qwen2Audio(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful assistant.'
        self.system_prompt_template = '<|im_start|>system\n{content}<|im_end|>'
        self.audio_prompt = 'Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>'
        self.user_prompt_template = '<|im_start|>user\n{content}<|im_end|>'
        self.assistant_prompt_template = '<|im_start|>assistant\n{content}<|im_end|>'
        self.generation_prefix = '<|im_start|>assistant\n'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = Qwen2AudioProcessor.from_pretrained(self.weight_path.lalm_path).tokenizer
        self.tokenizer.padding_side = 'left'
        self.llm_base = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.weight_path.lalm_path,
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        self.im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids('<|audio_bos|>')
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids('<|audio_eos|>')
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<|AUDIO|>')
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
    
    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def expand_prompt(self, prompts, audio_lengths):
        audio_token = '<|AUDIO|>'
        expanded_prompts = []
        for prompt in prompts:
            replace_str = []
            while audio_token in prompt:
                audio_length = audio_lengths.pop(0)
                input_length = (audio_length - 1) // 2 + 1
                num_audio_tokens = (input_length - 2) // 2 + 1
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
        input_features, feature_attention_mask = None, None
        if len(audio_inputs) > 0:
            input_features, feature_attention_mask = self.audio_processor(audio_inputs)
            fbank_token_mask = feature_attention_mask[:, ::4]
            feature_attention_mask = feature_attention_mask.bool()
            input_features = input_features.to(dtype=self.dtype)
            audio_lengths = feature_attention_mask.sum(-1).tolist()
            text_inputs = self.expand_prompt(text_inputs, audio_lengths)
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'input_features': input_features,
            'feature_attention_mask': feature_attention_mask,
        }
        input_length = input_ids.size(1)
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            im_start_idx = (input_id == self.im_start_id).nonzero(as_tuple=True)[0].cpu().tolist()
            im_end_idx = (input_id == self.im_end_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(im_end_idx) > 0 and len(im_start_idx) > 0:
                for b, e in zip(im_start_idx[1:-1], im_end_idx[1:]):
                    input_mask[i, b+3:e] = 2 # text user prompt mask: 2
                input_mask[i, im_start_idx[0]+3:im_end_idx[0]] = 3 # system / function prompt mask: 3
                input_mask[i, im_start_idx[-1]+3:] = 4 # target index mask: 4
            if len(audio_inputs) > 0:
                audio_idx = (input_id == self.audio_token_id).nonzero(as_tuple=True)[0]
                if audio_idx.numel() > 0:
                    input_mask[i, audio_idx] = fbank_token_mask[i, :len(audio_idx)]
        input_mask[(input_ids == self.audio_start_id) | (input_ids == self.audio_end_id)] = 0 # special token mask: 0
        return input_params, input_length, input_mask