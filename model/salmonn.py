import torch
import torch.nn.functional as F
from transformers import WhisperModel, LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram, pad_or_trim_list, pad_or_trim_tensor
from model.salmonn_src import BEATsConfig, BEATs, BertConfig, BertLMHeadModel


@LALMFactory.register('salmonn')
class SALMONN(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = ''
        self.system_prompt_template = '{content}'
        self.audio_prompt = '<Speech><SpeechHere></Speech>'
        self.user_prompt_template = 'USER:{content}\n'
        self.assistant_prompt_template = 'ASSISTANT:{content}\n'
        self.generation_prefix = 'ASSISTANT:'
        
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.speech_encoder = WhisperModel.from_pretrained(
            self.weight_path.whisper_encoder_path,
            device_map=device,
            torch_dtype=dtype
        ).encoder
        self.ln_speech = torch.nn.LayerNorm(self.speech_encoder.config.d_model)
        beats_ckpt = torch.load(self.weight_path.beats_encoder_path, weights_only=True, map_location=device)
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        self.beats = BEATs(beats_cfg)
        self.beats.load_state_dict(beats_ckpt['model'])
        self.ln_audio = torch.nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.weight_path.llm_path, use_fast=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "left"
        self.llm_base = LlamaForCausalLM.from_pretrained(
            self.weight_path.llm_path,
            _attn_implementation='eager',
            device_map=device,
            torch_dtype=dtype)
        self.llm_base.resize_token_embeddings(len(self.tokenizer))
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=self.parameter.lora_rank, 
            lora_alpha=self.parameter.lora_alpha, lora_dropout=self.parameter.lora_dropout)
        self.llm_base = get_peft_model(self.llm_base, self.peft_config)
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = 2
        encoder_config.encoder_width = self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
        encoder_config.is_decoder = True
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = self.parameter.num_speech_query_token
        self.speech_Qformer = BertLMHeadModel(config=encoder_config)
        self.speech_Qformer.cls = None
        self.speech_Qformer.bert.embeddings.word_embeddings = None
        self.speech_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.speech_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.speech_query_tokens = torch.nn.Parameter(
            torch.zeros(1, self.parameter.num_speech_query_token, encoder_config.hidden_size)
        )
        self.speech_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        self.speech_llama_proj = torch.nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llm_base.config.hidden_size
        )
        ckpt = torch.load(self.weight_path.lalm_path, weights_only=True, map_location=device)
        self.load_state_dict(ckpt['model'], strict=False)
        self.eval()
        self.to(device, dtype)
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
        
    def batch_encode(self, texts, **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt', **kwargs)
    
    def batch_decode(self, ids):
        return self.tokenizer.batch_decode(ids, add_special_tokens=False, skip_special_tokens=True)
    
    def get_embeds(self, input_ids):
        return self.llm_base.model.model.embed_tokens(input_ids)
    
    def extract_audio_embeds(self, audios):
        fbanks, _ = self.audio_processor(audios)
        fbanks = fbanks.to(dtype=self.dtype)
        speech_embs = self.speech_encoder(fbanks, return_dict=True).last_hidden_state
        speech_embs = self.ln_speech(speech_embs)
        pad_func = pad_or_trim_list if isinstance(audios, list) else pad_or_trim_tensor
        audios, _ = pad_func(audios, self.fbank_config.n_samples)
        audio_embs, _ = self.beats.extract_features(audios, padding_mask=None, feature_only=True)
        audio_embs = self.ln_audio(audio_embs)
        if audio_embs.size(1) < speech_embs.size(1):
            audio_embs = F.pad(audio_embs, (0, 0, 0, speech_embs.size(1) - audio_embs.size(1)))
        elif audio_embs.size(1) > speech_embs.size(1):
            speech_embs = F.pad(speech_embs, (0, 0, 0, audio_embs.size(1) - speech_embs.size(1)))
        speech_embs = torch.cat((speech_embs, audio_embs), dim=-1)
        B, T, C = speech_embs.shape
        kernel = (1, round(1500 * self.parameter.second_per_window / 30.0))
        stride = (1, round(1500 * self.parameter.second_stride / 30.0))
        speech_embs_tr = speech_embs.transpose(1, 2).unsqueeze(2)
        speech_embs_overlap = F.unfold(speech_embs_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        L = speech_embs_overlap.shape[2]
        speech_embs_overlap = speech_embs_overlap.view(B, -1, kernel[1], L)
        speech_embs_overlap = torch.permute(speech_embs_overlap, [0, 3, 2, 1])
        speech_embs = speech_embs_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embs.size()[:-1], dtype=torch.long, device=speech_embs.device)
        query_tokens = self.speech_query_tokens.expand(speech_embs.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embs,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embs = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embs = speech_embs.view(B, -1, speech_embs.size(2)).contiguous()
        speech_atts = torch.ones(speech_embs.size()[:-1], dtype=torch.long).to(speech_embs.device)
        return speech_embs, speech_atts
    
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + ' ' + label for text, label in zip(text_inputs, labels)]
            label_token = self.tokenizer.encode(labels[0], add_special_tokens=False)
        if len(audio_inputs) > 0:
            audio_embeds, audio_attn_mask = self.extract_audio_embeds(audio_inputs)
            text_inputs = [text_input.split("<SpeechHere>") for text_input in text_inputs]
            before_inputs = self.batch_encode([prompt[0] for prompt in text_inputs])
            before_embeds = self.get_embeds(before_inputs['input_ids'].to(self.device))
            before_attn_mask = before_inputs['attention_mask'].to(self.device)
            after_inputs = self.batch_encode([text_input[1] for text_input in text_inputs], add_special_tokens=False)
            after_embeds = self.get_embeds(after_inputs['input_ids'].to(self.device))
            after_attn_mask = after_inputs['attention_mask'].to(self.device)
            input_embeds = torch.cat([before_embeds, audio_embeds, after_embeds], dim=1)
            attention_mask = torch.cat([before_attn_mask, audio_attn_mask, after_attn_mask], dim=1)
        else:
            inputs = self.batch_encode(text_inputs)
            input_embeds = self.get_embeds(inputs['input_ids'].to(self.device))
            attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
        }
        input_length = 0
        input_mask = torch.zeros_like(attention_mask)
        if len(audio_inputs) > 0:
            input_mask[:, before_embeds.size(1):-after_embeds.size(1)] = audio_attn_mask
        if labels is not None and len(labels) == len(text_inputs):
            input_mask[:, -len(label_token):] = 4
        return input_params, input_length, input_mask