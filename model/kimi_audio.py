import torch
from transformers import WhisperModel
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.glm4_voice_src import WhisperVQEncoder
from model.kimi_audio_src import TikTokenTokenizer, MoonshotKimiaForCausalLM
from model.kimi_audio_src.special_tokens import instantiate_extra_tokens, extra_tokens_tolist


@LALMFactory.register('kimi_audio')
class KimiAudio(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = ''
        self.system_prompt_template = '{content}'
        self.audio_prompt = '<|im_media_begin|>{audio_prompt}<|im_media_end|><|im_kimia_speech_ct_id|>'
        self.user_prompt_template = '<|im_kimia_user_msg_start|>{content}<|im_msg_end|>'
        self.assistant_prompt_template = '<|im_kimia_assistant_msg_start|>{content}<|im_msg_end|>'
        self.generation_prefix = '<|im_kimia_assistant_msg_start|>\n'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.audio_encoder = WhisperModel.from_pretrained(
            self.weight_path.encoder_path,
            device_map=device,
            torch_dtype=dtype
        ).encoder
        self.audio_tokenizer = WhisperVQEncoder.from_pretrained(
            self.weight_path.tokenizer_path,
            device_map=device,
            torch_dtype=dtype)
        stride1 = self.audio_tokenizer.conv1.stride[0] * self.audio_tokenizer.conv2.stride[0]
        stride2 = self.audio_tokenizer.config.pooling_kernel_size or 1
        self.audio_stride = stride1 * stride2
        self.tokenizer = TikTokenTokenizer.from_pretrained(
            self.weight_path.lalm_path)
        self.llm_base = MoonshotKimiaForCausalLM.from_pretrained(
            self.weight_path.lalm_path,
            _attn_implementation='flash_attention_2',
            device_map=device,
            torch_dtype=dtype)
        self.embed_tokens = self.llm_base.get_input_embeddings()
        self.extra_tokens = instantiate_extra_tokens(self.tokenizer)
        self.eod_ids = extra_tokens_tolist(self.extra_tokens)
        self.llm_base.extra_tokens = self.extra_tokens
        self.audio_start = self.extra_tokens.media_begin
        self.audio_end = self.extra_tokens.media_end
        self.audio_first = self.llm_base.config.kimia_token_offset
        self.audio_last = self.llm_base.config.vocab_size - 1
        self.all_audio_embeds = self.embed_tokens.weight[self.audio_first:self.audio_last+1]
        self.special_ids = torch.LongTensor([self.audio_start, self.audio_end, self.extra_tokens.kimia_speech_ct_id]).to(self.device)
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
        
    def batch_encode(self, texts, **kwargs):
        token_ids = [self.tokenizer.encode(t, allowed_special='all', bos=False, eos=False) for t in texts]
        max_len = max(len(ids) for ids in token_ids)
        input_ids = []
        attention_mask = []
        for ids in token_ids:
            pad_len = max_len - len(ids)
            padded = [self.extra_tokens.pad] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            input_ids.append(padded)
            attention_mask.append(mask)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    def batch_decode(self, ids):
        # filter out audio token ids and eos token id
        text_ids = ids.tolist()
        text_ids = [[x for x in text_id if x < self.audio_first and x not in self.eod_ids] for text_id in text_ids]
        return [self.tokenizer.decode(text_id) for text_id in text_ids]
    
    def encode_audios(self, fbanks, fbank_mask, fbank_token_mask):
        # discrete audio token
        distances = self.audio_tokenizer(fbanks, fbank_mask).distances
        # continuous audio feat
        audio_feats_padded = self.audio_encoder(fbanks, return_dict=True).last_hidden_state
        assert fbank_token_mask.shape == distances.shape[:2]
        audio_tokens, audio_ids, audio_embeds, audio_feats = [], [], [], []
        for i in range(len(distances)):
            distance = distances[i][fbank_token_mask[i]]
            # gradient approximation using Gumbel-Softmax sampling with straight-through trick
            gumbel_onehot = torch.nn.functional.gumbel_softmax(-distance, tau=10.0, hard=True)
            audio_embed = gumbel_onehot @ self.all_audio_embeds
            audio_id = distance.argmin(dim=-1).detach() + self.audio_first
            # dummy audio tokens for batch encode
            audio_token = ['<|im_kimia_text_blank|>'] * len(audio_id)
            audio_feat = audio_feats_padded[i][:len(audio_token)*4]
            audio_feat = audio_feat.reshape(len(audio_token), -1)
            audio_tokens.append(audio_token)
            audio_ids.append(audio_id)
            audio_embeds.append(audio_embed)
            audio_feats.append(audio_feat)
        return audio_tokens, audio_ids, audio_embeds, audio_feats
    
    def split_input_ids(self, input_ids, attention_mask, audio_ids):
        text_input_ids = torch.ones_like(input_ids) * self.extra_tokens.kimia_text_blank
        audio_feat_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        text_start_index = (input_ids == self.extra_tokens.kimia_user_msg_start).nonzero(as_tuple=True)[1].tolist()
        audio_start_index = (input_ids == self.audio_start).nonzero(as_tuple=True)[1].tolist()
        audio_end_index = (input_ids == self.audio_end).nonzero(as_tuple=True)[1].tolist()
        for i, (t_s, a_s, a_e) in enumerate(zip(text_start_index, audio_start_index, audio_end_index)):
            text_input_ids[i, t_s+1:a_s] = input_ids[i, t_s+1:a_s]
            input_ids[i, t_s+1:a_s] = self.extra_tokens.kimia_text_blank
            # insert true audio ids here
            input_ids[i, a_s+1:a_e] = audio_ids[i]
            audio_feat_mask[i, a_s+1:a_e] = True
        return input_ids, text_input_ids, audio_feat_mask
        
    def encode_inputs(self, input_ids, audio_embeds, audio_feats, audio_feat_mask, text_input_ids):
        inputs_embeds = self.embed_tokens(input_ids).to(self.device)
        expand_feats = torch.zeros(
            inputs_embeds.shape[0], inputs_embeds.shape[1], audio_feats[0].shape[-1],
            device=self.device, dtype=audio_feats[0].dtype)
        start_index = (input_ids == self.audio_start).nonzero(as_tuple=True)[1].tolist()
        end_index = (input_ids == self.audio_end).nonzero(as_tuple=True)[1].tolist()
        assert len(start_index) == len(end_index), 'start and end index mismatch.'
        for i, (s, e) in enumerate(zip(start_index, end_index)):
            assert (e - s - 1) == audio_embeds[i].shape[0], f'emb length mismatch: {e-s-1} -> {audio_embeds[i].shape[0]}.'
            inputs_embeds[i, s+1:e] = audio_embeds[i]
            assert (e - s - 1) == audio_feats[i].shape[0], f'feat length mismatch: {e-s-1} -> {audio_feats[i].shape[0]}.'
            expand_feats[i, s+1:e] = audio_feats[i]
        expand_feats = expand_feats.transpose(0, 1)
        conti_audio_embeds = self.llm_base.model.vq_adaptor(expand_feats).transpose(0, 1)
        conti_audio_embeds = conti_audio_embeds * audio_feat_mask[:, :, None]
        scale_factor = torch.sqrt(torch.tensor(2.0, dtype=inputs_embeds.dtype, device=self.device))
        addwith_embeds = (inputs_embeds + conti_audio_embeds) * scale_factor
        inputs_embeds = inputs_embeds * (~audio_feat_mask[:, :, None]) + addwith_embeds * audio_feat_mask[:, :, None]
        inputs_embeds = inputs_embeds + self.embed_tokens(text_input_ids)
        return inputs_embeds
    
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        if len(audio_inputs) > 0:
            fbanks, fbank_mask = self.audio_processor(audio_inputs)
            fbanks = fbanks.to(dtype=self.dtype)
            fbank_token_mask = fbank_mask[:, ::self.audio_stride]
            audio_tokens, audio_ids, audio_embeds, audio_feats = self.encode_audios(fbanks, fbank_mask.bool(), fbank_token_mask.bool())
            audio_prompts = [''.join(audio_token) for audio_token in audio_tokens]
            text_inputs = [text_input.format(audio_prompt=audio_prompt) for text_input, audio_prompt in zip(text_inputs, audio_prompts)]
        inputs = self.batch_encode(text_inputs)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        if len(audio_inputs) > 0:
            input_ids, text_input_ids, audio_feat_mask = self.split_input_ids(input_ids, attention_mask, audio_ids)
            inputs_embeds = self.encode_inputs(input_ids, audio_embeds, audio_feats, audio_feat_mask, text_input_ids)
        else:
            inputs_embeds = self.embed_tokens(input_ids).to(self.device)
        input_params = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        input_length = 0
        input_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            text_start_idx = (
                (input_id == self.extra_tokens.kimia_user_msg_start) | 
                (input_id == self.extra_tokens.kimia_assistant_msg_start)
            ).nonzero(as_tuple=True)[0].cpu().tolist()
            text_end_idx = (input_id == self.extra_tokens.msg_end).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(text_start_idx) > 0 and len(text_end_idx) > 0:
                for b, e in zip(text_start_idx[:-1], text_end_idx):
                    input_mask[i, b+1:e] = 2 # text user prompt mask: 2
                input_mask[i, text_start_idx[-1]+2:] = 4 # target index mask: 4
            if len(audio_inputs) > 0:
                audio_idx = audio_feat_mask[i].nonzero(as_tuple=True)[0]
                if audio_idx.numel() > 0:
                    input_mask[i, audio_idx] = fbank_token_mask[i, :len(audio_idx)]
        input_mask[torch.isin(input_ids, self.special_ids)] = 0
        return input_params, input_length, input_mask
    