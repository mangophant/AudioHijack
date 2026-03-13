import torch
from transformers import AutoTokenizer
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.glm4_voice_src import WhisperVQEncoder
from model.vita_audio_src import Qwen2MTPForCausalLM


@LALMFactory.register('vita_audio')
class VitaAudio(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.'
        self.system_prompt_template = '<|im_start|>system\n{content}<|im_end|>'
        self.audio_prompt = '<|begin_of_audio|>{audio_prompt}<|end_of_audio|>'
        self.user_prompt_template = '<|im_start|>user\n{content}<|im_end|>'
        self.assistant_prompt_template = '<|im_start|>assistant\n{content}<|im_end|>'
        self.generation_prefix = '<|im_start|>assistant\n'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.audio_tokenizer = WhisperVQEncoder.from_pretrained(
            self.weight_path.tokenizer_path,
            device_map=device,
            torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path.lalm_path)
        self.tokenizer.padding_side = 'left'
        self.llm_base = Qwen2MTPForCausalLM.from_pretrained(
            self.weight_path.lalm_path,
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        # audio tokens: from "151685": "<|audio_0|>" to "168068": "<|audio_16383|>"
        self.im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.audio_start = self.tokenizer.convert_tokens_to_ids('<|begin_of_audio|>')
        self.audio_end = self.tokenizer.convert_tokens_to_ids('<|end_of_audio|>')
        self.audio_first = self.tokenizer.convert_tokens_to_ids('<|audio_0|>')
        self.audio_last = self.tokenizer.convert_tokens_to_ids('<|audio_16383|>')
        self.embed_tokens = self.llm_base.get_input_embeddings()
        self.all_audio_embeds = self.embed_tokens.weight[self.audio_first:self.audio_last+1]
        self.audio_output_pad_ids = self.tokenizer.encode(
            '<|begin_of_audio|>' + '<|audio_0|>' * 8 + '<|end_of_audio|>'
        )
        self.audio_output_pad_ids = torch.LongTensor(self.audio_output_pad_ids)
        self.audio_token_stride = self.audio_tokenizer.conv1.stride[0] * self.audio_tokenizer.conv2.stride[0]
        self.audio_token_stride *= self.audio_tokenizer.config.pooling_kernel_size or 1
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
        
    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        # filter out audio token ids
        text_ids = ids.clone()
        text_ids[(text_ids >= self.audio_first) & (text_ids <= self.audio_last)] = self.tokenizer.pad_token_id
        return self.tokenizer.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def encode_audios(self, fbanks, fbank_mask, fbank_token_mask):
        distances = self.audio_tokenizer(fbanks, fbank_mask).distances
        assert fbank_token_mask.shape == distances.shape[:-1]
        audio_tokens, audio_embeds = [], []
        for i in range(len(distances)):
            distance = distances[i][fbank_token_mask[i]]
            # gradient approximation using Gumbel-Softmax sampling with straight-through trick
            gumbel_onehot = torch.nn.functional.gumbel_softmax(-distance, tau=10.0, hard=True)
            audio_embed = gumbel_onehot @ self.all_audio_embeds
            audio_embeds.append(audio_embed)
            audio_codes = distance.argmin(dim=-1).detach().cpu().tolist()
            audio_tokens.append([f'<|audio_{str(code)}|>' for code in audio_codes])
        return audio_tokens, audio_embeds
    
    def encode_inputs(self, input_ids, audio_embeds):
        inputs_embeds = self.embed_tokens(input_ids)
        start_index = (input_ids == self.audio_start).nonzero(as_tuple=True)[1].tolist()
        end_index = (input_ids == self.audio_end).nonzero(as_tuple=True)[1].tolist()
        assert len(start_index) == len(end_index), 'start and end index mismatch.'
        for i, (s, e) in enumerate(zip(start_index, end_index)):
            assert (e - s - 1) == audio_embeds[i].shape[0], f'length mismatch: {e-s-1} -> {audio_embeds[i].shape[0]}.'
            inputs_embeds[i, s+1:e] = audio_embeds[i]
        return inputs_embeds
    
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if len(audio_inputs) > 0:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(dtype=self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_token_stride]
            audio_tokens, audio_embeds = self.encode_audios(fbanks, fbank_mask.bool(), fbank_token_mask.bool())
            audio_prompts = [''.join(audio_token) for audio_token in audio_tokens]
            text_inputs = [text_input.format(audio_prompt=audio_prompt) for text_input, audio_prompt in zip(text_inputs, audio_prompts)]
            inputs = self.batch_encode(text_inputs)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            inputs_embeds = self.encode_inputs(input_ids, audio_embeds)
        else:
            inputs = self.batch_encode(text_inputs)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
        if labels is not None:
            target_ids, target_mask = self.prepare_output(labels)
            input_ids = torch.hstack((input_ids, target_ids))
            attention_mask = torch.hstack((attention_mask, torch.ones_like(target_mask)))
            if inputs_embeds is not None:
                target_embeds = self.embed_tokens(target_ids)
                inputs_embeds = torch.cat((inputs_embeds, target_embeds), dim=1)
        if mode == 'forward' and inputs_embeds is not None:
            input_params = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
            }
        else:
            input_params = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
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
                if labels is not None:
                    input_mask[i, im_start_idx[-1]+3:] = input_mask[i, im_start_idx[-1]+3:].masked_fill(target_mask[i], 4) # target index mask: 4
            audio_start_idx = (input_id == self.audio_start).nonzero(as_tuple=True)[0].cpu().tolist()
            audio_end_idx = (input_id == self.audio_end).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(audio_end_idx) > 0 and len(audio_start_idx) > 0:
                input_mask[i, audio_start_idx[0]+1:audio_end_idx[0]] = fbank_token_mask[i, :audio_end_idx[0]-audio_start_idx[0]-1]
        input_mask[(input_ids == self.audio_start) | (input_ids == self.audio_end)] = 0 # special token mask: 0
        return input_params, input_length, input_mask
            
    def pack_output(self, ids, mask):
        start, interval = 1, 4
        pad = self.audio_output_pad_ids.repeat((ids.shape[0], 1))
        pad_mask = torch.tensor([False] * pad.shape[-1]).bool()
        pad_mask = pad_mask.repeat((ids.shape[0], 1))
        packed_ids = [ids[:, 0:start], pad]
        packed_mask = [mask[:, 0:start], pad_mask]
        idx = start
        while idx + interval < ids.shape[-1]:
            packed_ids.append(ids[:, idx:idx+interval])
            packed_mask.append(mask[:, idx:idx+interval])
            packed_ids.append(pad)
            packed_mask.append(pad_mask)
            idx += interval
        if idx < ids.shape[-1]:
            packed_ids.append(ids[:, idx:])
            packed_mask.append(mask[:, idx:])
        packed_ids = torch.hstack(packed_ids).to(self.device)
        packed_mask = torch.hstack(packed_mask).to(self.device)
        return packed_ids, packed_mask