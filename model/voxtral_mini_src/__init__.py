from transformers import AutoConfig, AutoModel
from .configuration_voxtral import VoxtralConfig, VoxtralEncoderConfig
from .tokenization_voxtral import MistralCommonTokenizer
from .modeling_voxtral import VoxtralForConditionalGeneration, VoxtralEncoder


AutoConfig.register("voxtral_encoder", VoxtralEncoderConfig)
AutoModel.register(VoxtralEncoderConfig, VoxtralEncoder)
AutoConfig.register("voxtral", VoxtralConfig)
AutoModel.register(VoxtralConfig, VoxtralForConditionalGeneration)