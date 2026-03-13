from transformers import AutoConfig, AutoProcessor, AutoModel
from .ultravox_config import UltravoxConfig
from .ultravox_processing import UltravoxProcessor
from .ultravox_model import UltravoxModel


AutoConfig.register('ultravox', UltravoxConfig)
AutoProcessor.register(UltravoxConfig, UltravoxProcessor)
AutoModel.register(UltravoxConfig, UltravoxModel)