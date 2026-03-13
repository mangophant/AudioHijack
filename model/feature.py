import torch
from transformers.audio_utils import mel_filter_bank
from model.phi4_mini_src import speechlib_mel


def pad_or_trim_list(audios, length):
    audios_data, attention_mask = [], []
    for audio in audios:
        # attention_mask: pad -> 0 audio data -> 1, speech prompt -> 2
        if isinstance(audio, dict):
            audio_data, speech_prompt, is_audio_first = \
                audio['audio_data'], audio['speech_prompt'], audio['is_audio_first']
            audio_data_len, speech_prompt_len = audio_data.shape[-1], speech_prompt.shape[-1]
            attention_mask_i = torch.ones(length, device=audio_data.device).long()
            if is_audio_first:
                audio = torch.cat((audio_data, speech_prompt), dim=-1)
                attention_mask_i[audio_data_len:audio_data_len+speech_prompt_len] = 2
            else:
                audio = torch.cat((speech_prompt, audio_data), dim=-1)
                attention_mask_i[:speech_prompt_len] = 2
        else:
            attention_mask_i = torch.ones(length, device=audio.device).long()
        if audio.shape[-1] >= length:
            audio_data = audio.index_select(dim=-1, index=torch.arange(length, device=audio.device))
        elif audio.shape[-1] < length:
            pad_widths = [(0, 0)] * audio.ndim
            pad_widths[-1] = (0, length - audio.shape[-1])
            audio_data = torch.nn.functional.pad(audio, [pad for sizes in pad_widths[::-1] for pad in sizes])
            attention_mask_i[audio.shape[-1]:] = 0
        audios_data.append(audio_data)
        attention_mask.append(attention_mask_i)
    return torch.vstack(audios_data), torch.vstack(attention_mask)


def pad_or_trim_tensor(audios, length):
    attention_mask = torch.ones((audios.shape[0], length), device=audios.device).long()
    if audios.shape[-1] >= length:
        audios_data = audios.index_select(dim=-1, index=torch.arange(length, device=audios.device))
    elif audios.shape[-1] < length:
        pad_widths = [(0, 0)] * audios.ndim
        pad_widths[-1] = (0, length - audios.shape[-1])
        audios_data = torch.nn.functional.pad(audios, [pad for sizes in pad_widths[::-1] for pad in sizes])
        attention_mask[..., audios.shape[-1]:] = 0
    return audios_data, attention_mask


class LogMelSpectrogram(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.config.n_fft // 2,
            num_mel_filters=self.config.n_mels,
            min_frequency=self.config.min_frequency,
            max_frequency=self.config.max_frequency,
            sampling_rate=self.config.sample_rate,
            norm='slaney',
            mel_scale='slaney',
        )
        self.mel_filters = torch.from_numpy(self.mel_filters).float()
        self.window = torch.hann_window(self.config.win_length).float()
        self.lfr_m = getattr(config, 'lfr_m', None)
        self.lfr_n = getattr(config, 'lfr_n', None)
        
    def to(self, device):
        self.mel_filters = self.mel_filters.to(device)
        self.window = self.window.to(device)
        
    def apply_lfr(self, fbank, attn_mask):
        B, D, T = fbank.shape
        T_out = (T - self.lfr_m) // self.lfr_n + 1
        lfr_features, lfr_masks = [], []
        for i in range(T_out):
            start = i * self.lfr_n
            end = start + self.lfr_m
            stacked = fbank[:, :, start:end].reshape(B, D * self.lfr_m)
            lfr_features.append(stacked)
            mask_chunk = attn_mask[:, start:end]
            all_nonzero = (mask_chunk != 0).all(dim=1).long()
            valid = mask_chunk[:, 0] * all_nonzero
            lfr_masks.append( valid)
        lfr_mask = torch.stack(lfr_masks, dim=1)
        lfr_fbank = torch.stack(lfr_features, dim=2)
        return lfr_fbank, lfr_mask

    def forward(self, audios):
        audios, attention_mask = pad_or_trim_list(audios, self.config.n_samples)
        stft = torch.stft(audios, self.config.n_fft, self.config.hop_length, win_length=self.config.win_length, window=self.window, return_complex=True)
        spec = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T @ spec
        log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        attention_mask = attention_mask[:, ::self.config.hop_length]
        if self.lfr_m and self.lfr_n:
            return self.apply_lfr(log_spec, attention_mask)
        return log_spec, attention_mask

    
class LogMelSpectrogramE(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel_filters = speechlib_mel(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.min_frequency,
            fmax=self.config.max_frequency
        )
        self.mel_filters = torch.from_numpy(self.mel_filters).float()
        self.window = torch.hamming_window(self.config.win_length).float()
        
    def to(self, device):
        self.mel_filters = self.mel_filters.to(device)
        self.window = self.window.to(device)

    def forward(self, audios):
        audios, attention_mask = pad_or_trim_list(audios, self.config.n_samples)
        y_frames = audios.unfold(dimension=1, size=self.config.win_length, step=self.config.hop_length)
        y_frames_prev = torch.roll(y_frames, shifts=1, dims=2)
        y_frames_prev[..., 0] = y_frames[..., 1]
        y_frames = (y_frames - self.config.preemphasis * y_frames_prev) * 32768.0
        S = torch.fft.rfft(self.window * y_frames, n=self.config.n_fft, dim=2).to(torch.complex64)
        spec = S.abs().float() ** 2
        mel_spec = spec @ self.mel_filters.T
        log_spec = torch.log(torch.clamp(mel_spec, min=1.0)).transpose(1, 2)
        attention_mask = attention_mask[:, ::self.config.hop_length][:, :log_spec.shape[-1]]
        return log_spec, attention_mask