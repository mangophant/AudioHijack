from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_qwen2 import Qwen2MTPConfig
from .modeling_qwen2 import Qwen2MTPForCausalLM


AutoConfig.register("qwen2_mtp", Qwen2MTPConfig)
AutoModelForCausalLM.register(Qwen2MTPConfig, Qwen2MTPForCausalLM)
