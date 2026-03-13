from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from .configuration_moonshot_kimia import KimiAudioConfig
from .tokenization_kimia import TikTokenTokenizer
from .modeling_moonshot_kimia import MoonshotKimiaModel, MoonshotKimiaForCausalLM
from .sampler import KimiASampler
from .special_tokens import instantiate_extra_tokens, extra_tokens_tolist


AutoConfig.register("kimi_audio", KimiAudioConfig)
AutoTokenizer.register(KimiAudioConfig, TikTokenTokenizer)
AutoModel.register(KimiAudioConfig, MoonshotKimiaModel)
AutoModelForCausalLM.register(KimiAudioConfig, MoonshotKimiaForCausalLM)
