import base64
import io
import os
import librosa
import soxr
import requests
import soundfile as sf

from typing import Any, Optional, Union
from transformers.utils import requires_backends




def load_audio_as(
    audio: str,
    return_format: str,
    timeout: Optional[int] = None,
    force_mono: bool = False,
    sampling_rate: Optional[int] = None,
) -> Union[str, dict[str, Any], io.BytesIO, None]:
    """
    Load audio from either a local file path or URL and return in specified format.

    Args:
        audio (`str`): Either a local file path or a URL to an audio file
        return_format (`str`): Format to return the audio in:
            - "base64": Base64 encoded string
            - "dict": Dictionary with data and format
            - "buffer": BytesIO object
        timeout (`int`, *optional*): Timeout for URL requests in seconds
        force_mono (`bool`): Whether to convert stereo audio to mono
        sampling_rate (`int`, *optional*): If provided, the audio will be resampled to the specified sampling rate.

    Returns:
        `Union[str, Dict[str, Any], io.BytesIO, None]`:
            - `str`: Base64 encoded audio data (if return_format="base64")
            - `dict`: Dictionary with 'data' (base64 encoded audio data) and 'format' keys (if return_format="dict")
            - `io.BytesIO`: BytesIO object containing audio data (if return_format="buffer")
    """
    # TODO: @eustlb, we actually don't need librosa but soxr is installed with librosa
    requires_backends(load_audio_as, ["librosa"])

    if return_format not in ["base64", "dict", "buffer"]:
        raise ValueError(f"Invalid return_format: {return_format}. Must be 'base64', 'dict', or 'buffer'")

    try:
        # Load audio bytes from URL or file
        audio_bytes = None
        if audio.startswith(("http://", "https://")):
            response = requests.get(audio, timeout=timeout)
            response.raise_for_status()
            audio_bytes = response.content
        elif os.path.isfile(audio):
            with open(audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        else:
            raise ValueError(f"File not found: {audio}")

        # Process audio data
        with io.BytesIO(audio_bytes) as audio_file:
            with sf.SoundFile(audio_file) as f:
                audio_array = f.read(dtype="float32")
                original_sr = f.samplerate
                audio_format = f.format
                if sampling_rate is not None and sampling_rate != original_sr:
                    # Resample audio to target sampling rate
                    audio_array = soxr.resample(audio_array, original_sr, sampling_rate, quality="HQ")
                else:
                    sampling_rate = original_sr

        # Convert to mono if needed
        if force_mono and audio_array.ndim != 1:
            audio_array = audio_array.mean(axis=1)

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sampling_rate, format=audio_format.upper())
        buffer.seek(0)

        if return_format == "buffer":
            return buffer
        elif return_format == "base64":
            return base64.b64encode(buffer.read()).decode("utf-8")
        elif return_format == "dict":
            return {
                "data": base64.b64encode(buffer.read()).decode("utf-8"),
                "format": audio_format.lower(),
            }

    except Exception as e:
        raise ValueError(f"Error loading audio: {e}")
