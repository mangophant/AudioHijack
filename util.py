import os
import io
import base64
import shutil
import random
import torch
import soundfile
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from pesq import pesq
from pystoi import stoi
from mel_cepstral_distance import compare_audio_files
from transformers import set_seed as tfm_set_seed


def set_dir(path, recreate=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif recreate:
        shutil.rmtree(path)
        os.makedirs(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    tfm_set_seed(seed)
    
    
def set_device_dtype(gpu, device, bf16):
    if gpu:
        device = torch.device(f'cuda:{device}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    dtype = torch.bfloat16 if bf16 else torch.float32
    return device, dtype


def check_audio_length(example, min_len=5, max_len=20):
    audio = example["audio"]
    length, sr = len(audio["array"]), audio["sampling_rate"]
    return length >= min_len * sr and length <= max_len * sr


def load_audio(path, min_len=5, max_len=30, sr=16000):
    audio, sr = librosa.load(path, sr=sr)
    audio = torch.from_numpy(audio[:sr*max_len])
    if len(audio) < sr * min_len:
        audio = torch.hstack([audio, torch.zeros(sr * min_len - len(audio))])
    return audio


def save_audio(audio, path, sr=16000):
    audio = audio.squeeze(0).detach().cpu().numpy()
    soundfile.write(path, audio, sr)
    
    
def norm_audio(audio, epsilon=1e-8):
    max_vals = torch.max(torch.abs(audio)) * 2
    max_vals = torch.clamp(max_vals, min=epsilon)
    return audio / max_vals


def rms(sig):
    return torch.sqrt(torch.mean(sig**2))


def norm_reverbed_audio(audio, ref_audio):
    return audio / rms(audio) * rms(ref_audio)


def b64encode_audio_file(audio_file):
    audio_bytes = Path(audio_file).read_bytes()
    audio_bytes = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_bytes


def b64encode_audio(audio, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, audio.cpu().numpy().flatten(), sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def sample_batch(data, step, batch_size):
    start = (step * batch_size) % len(data)
    end = start + batch_size
    if end <= len(data):
        batch = data[start:end]
    else:
        batch = data[start:] + data[:end - len(data)]
    return batch
    

def calc_perceptual_metrics(ben_audio_file, adv_audio_file, sample_rate=16000, n_mfcc=13, eps=1e-10):
    ben_audio = load_audio(ben_audio_file).squeeze().numpy()
    adv_audio = load_audio(adv_audio_file).squeeze().numpy()
    # SNR
    signal_power = np.sum(ben_audio ** 2)
    noise_power = np.sum((ben_audio - adv_audio) ** 2)
    snr_score = 10 * np.log10(signal_power / (noise_power + eps) + eps)
    # MCD
    mcd_score, _ = compare_audio_files(ben_audio_file, adv_audio_file)
    # STOI & PESQ
    try:
        stoi_score = stoi(ben_audio, adv_audio, sample_rate, extended=False)
        pesq_score = pesq(sample_rate, ben_audio, adv_audio, 'wb')
    except Exception as e:
        stoi_score, pesq_score = 0, 0
        print(adv_audio_file, str(e))
    return snr_score, mcd_score, stoi_score, pesq_score


def calc_tool_call_metrics(call_success_list):
    if not call_success_list:
        return 0.0, 0.0, 0.0, 0.0
    total = len(call_success_list)
    invocations = [i for (i, _, _) in call_success_list]
    syntaxes = [s for (i, s, _) in call_success_list if i]
    executions = [e for (i, s, e) in call_success_list if i and s]
    invocation_success_rate = sum(invocations) / total
    syntax_success_rate = (sum(syntaxes) / len(syntaxes)) if syntaxes else 0.0
    execution_success_rate = (sum(executions) / len(executions)) if executions else 0.0
    call_success_rate = invocation_success_rate * syntax_success_rate * execution_success_rate
    return invocation_success_rate, syntax_success_rate, execution_success_rate, call_success_rate
