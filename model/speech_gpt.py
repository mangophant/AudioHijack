import os
import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from model.lalm import LALM, LALMFactory
from model.speech_gpt_src import Speech2Unit


@LALMFactory.register('speech_gpt')
class SpeechGPT(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University. SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n'
        self.system_prompt_template = '{content}'
        self.audio_prompt = '<sosp>{audio_prompt}<eosp>'
        self.user_prompt_template = '[Human]: {content}<eoh>. '
        self.assistant_prompt_template = '[SpeechGPT]: {content}<eoa>.'
        self.generation_prefix = '[SpeechGPT]: '
        # self.generation_prefix = '[SpeechGPT] : [ta] '
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_tokenizer = Speech2Unit(ckpt_dir=self.weight_path.tokenizer_path, device=device)
        cm_weight_path = os.path.join(self.weight_path.lalm_path, 'SpeechGPT-7B-cm')
        com_weight_path = os.path.join(self.weight_path.lalm_path, 'SpeechGPT-7B-com')
        self.tokenizer = LlamaTokenizer.from_pretrained(cm_weight_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.llm_base = LlamaForCausalLM.from_pretrained(
            cm_weight_path, 
            load_in_8bit=False, 
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        self.llm_base = PeftModel.from_pretrained(
            self.llm_base,
            com_weight_path,
            device_map=device,
            torch_dtype=dtype)
        # audio tokens: from <0>: 32000 to <999>: 32999
        # audio pad tokens: "<sosp>": 33000, "<eosp>": 33001
        self.audio_start = self.tokenizer.convert_tokens_to_ids('<sosp>')
        self.audio_end = self.tokenizer.convert_tokens_to_ids('<eosp>')
        self.audio_first = self.tokenizer.convert_tokens_to_ids('<0>')
        self.audio_last = self.tokenizer.convert_tokens_to_ids('<999>')
        self.embed_tokens = self.llm_base.get_input_embeddings()
        self.all_audio_embeds = self.embed_tokens.weight[self.audio_first:self.audio_last+1]
        self.user_token_id = self.tokenizer.convert_tokens_to_ids('[Human]')
        self.assist_token_id = self.tokenizer.convert_tokens_to_ids('[SpeechGPT]')
        self.eval()
        self.audio_tokenizer.requires_grad_(False)
        self.llm_base.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()

    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        text_ids = ids.clone()
        text_ids[(text_ids >= self.audio_first) & (text_ids <= self.audio_last)] = self.tokenizer.pad_token_id
        return self.tokenizer.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def encode_audios(self, audios):
        audio_tokens, audio_embeds, audio_token_mask = [], [], []
        for audio in audios:
            if isinstance(audio, dict):
                audio_data, speech_prompt, is_audio_first = \
                    audio['audio_data'], audio['speech_prompt'], audio['is_audio_first']
                audio_data_len, speech_prompt_len = audio_data.shape[-1], speech_prompt.shape[-1]
                if is_audio_first:
                    audio = torch.cat((audio_data, speech_prompt), dim=-1)
                else:
                    audio = torch.cat((speech_prompt, audio_data), dim=-1)
            else:
                speech_prompt_len, audio_data_len = 0, audio.shape[-1]
            feat = self.audio_tokenizer.feature_reader.get_feats(audio).to(dtype=self.dtype)
            _, distance = self.audio_tokenizer.apply_kmeans(feat)
            # gradient approximation using Gumbel-Softmax sampling with straight-through trick
            gumbel_onehot = torch.nn.functional.gumbel_softmax(-distance, tau=10.0, hard=True)
            audio_embed = gumbel_onehot @ self.all_audio_embeds
            audio_embeds.append(audio_embed)
            audio_codes = distance.argmin(dim=-1).detach().cpu().tolist()
            audio_tokens.append([f'<{str(code)}>' for code in audio_codes])
            token_mask = torch.ones(len(audio_codes), device=audio.device).long()
            speech_prompt_token_len = int(len(audio_codes) * speech_prompt_len / (speech_prompt_len + audio_data_len))
            if speech_prompt_token_len > 0:
                if self.is_audio_first:
                    token_mask[-speech_prompt_token_len:] = 2
                else:
                    token_mask[:speech_prompt_token_len] = 2
            audio_token_mask.append(token_mask)
        return audio_tokens, audio_embeds, audio_token_mask
    
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
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        if len(audio_inputs) > 0:
            audio_tokens, audio_embeds, audio_token_mask = self.encode_audios(audio_inputs)
            audio_prompts = [''.join(audio_token) for audio_token in audio_tokens]
            text_inputs = [text_input.format(audio_prompt=audio_prompt) for text_input, audio_prompt in zip(text_inputs, audio_prompts)]
            inputs = self.batch_encode(text_inputs)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            inputs_embeds = self.encode_inputs(input_ids, audio_embeds)
            input_params = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
            }
            input_length = 0
        else:
            inputs = self.batch_encode(text_inputs)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            input_params = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            input_length = input_ids.size(1)
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            user_token_idx = (input_id == self.user_token_id).nonzero(as_tuple=True)[0].cpu().tolist()
            assist_token_idx = (input_id == self.assist_token_id).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(user_token_idx) > 0 and len(assist_token_idx) > 0:
                for b, e in zip(user_token_idx, assist_token_idx):
                    input_mask[i, b+2:e-2] = 2 # text user prompt mask: 2
                input_mask[i, :user_token_idx[0]] = 3 # system / function prompt mask: 3
                input_mask[i, assist_token_idx[-1]+2:] = 4 # target index mask: 4
            audio_start_idx = (input_id == self.audio_start).nonzero(as_tuple=True)[0].cpu().tolist()
            audio_end_idx = (input_id == self.audio_end).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(audio_end_idx) > 0 and len(audio_start_idx) > 0:
                input_mask[i, audio_start_idx[0]+1:audio_end_idx[0]] = audio_token_mask[i]
        input_mask[(input_ids == self.audio_start) | (input_ids == self.audio_end)] = 0 # special token mask: 0
        input_mask[~attention_mask.bool()] = 0
        return input_params, input_length, input_mask