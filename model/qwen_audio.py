import torch
from model.lalm import LALM, LALMFactory
from model.feature import LogMelSpectrogram
from model.qwen_audio_src import QWenTokenizer, QWenLMHeadModel


@LALMFactory.register('qwen_audio')
class QwenAudio(LALM):
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = 'You are a helpful assistant.'
        self.system_prompt_template = '<|im_start|>system\n{content}<|im_end|>'
        self.audio_prompt = 'Audio 1: <audio>audio_url</audio>'
        self.user_prompt_template = '<|im_start|>user\n{content}<|im_end|>'
        self.assistant_prompt_template = '<|im_start|>assistant\n{content}<|im_end|>'
        self.generation_prefix = '<|im_start|>assistant\n'
    
    def load(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.audio_processor = LogMelSpectrogram(self.fbank_config)
        self.audio_processor.to(device)
        self.tokenizer = QWenTokenizer.from_pretrained(
            self.weight_path.lalm_path)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.llm_base = QWenLMHeadModel.from_pretrained(
            self.weight_path.lalm_path,
            use_flash_attn=False,
            device_map=device, 
            torch_dtype=dtype)
        self.im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids('<audio>')
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids('</audio>')
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('[[[AUDIO:modality]]]')
        self.eval()
        self.requires_grad_(False)
        self.llm_base.gradient_checkpointing_enable()
    
    def batch_encode(self, texts, **kwargs):
        audio_info = kwargs.pop("audio_info", [None]*len(texts))
        return self.tokenizer(texts, padding=True, return_tensors='pt', audio_info=audio_info, **kwargs)
    
    def batch_decode(self, ids):
        eos_id = torch.tensor([self.tokenizer.eod_id]).to(ids.device).unsqueeze(0).repeat(ids.size(0), 1)
        ids = torch.cat([ids, eos_id], dim=1)
        eos_id_pos = ids.eq(self.tokenizer.eod_id).float().argmax(-1)
        pred = [ids[i, :eos_id_pos[i]].tolist() for i in range(ids.size(0))]
        return [self.tokenizer.decode(_, skip_special_tokens=True).strip() for _ in pred]
    
    def get_T_after_cnn(self, L_in, dilation=1):
        for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out
    
    def get_audio_info(self, feature_len):
        audio_len_after_cnn = self.get_T_after_cnn(feature_len)
        audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
        input_audio_lengths = torch.tensor(
            [[audio_len_after_cnn, audio_token_num]], 
            dtype=torch.int, device=self.device
        )
        audio_span_tokens = [audio_token_num + 2]
        return input_audio_lengths, audio_span_tokens
    
    def pack_input(self, text_inputs, audio_inputs, labels, mode):
        if labels is not None:
            text_inputs = [text + label for text, label in zip(text_inputs, labels)]
        audio_info = None
        if len(audio_inputs) > 0:
            input_features, feature_attention_mask = self.audio_processor(audio_inputs)
            input_features = input_features.to(dtype=self.dtype)
            feature_attention_mask = feature_attention_mask.bool()
            len_info = [self.get_audio_info(mask.sum()) for mask in feature_attention_mask]
            audio_info = [{
                'input_audios': a.unsqueeze(0),
                'input_audio_lengths': b,
                'audio_span_tokens': c
                } for a, (b, c) in zip(input_features, len_info)]
        inputs = self.batch_encode(text_inputs, audio_info=audio_info)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        input_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio_info': audio_info,
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
                    input_mask[i, audio_idx] = 1
        input_mask[(input_ids == self.audio_start_id) | (input_ids == self.audio_end_id)] = 0 # special token mask: 0
        return input_params, input_length, input_mask
    