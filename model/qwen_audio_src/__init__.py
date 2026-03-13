from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from .configuration_qwen import QWenConfig
from .tokenization_qwen import QWenTokenizer
from .modeling_qwen import QWenLMHeadModel


AutoConfig.register("qwen", QWenConfig)
AutoTokenizer.register(QWenConfig, QWenTokenizer)
AutoModelForCausalLM.register(QWenConfig, QWenLMHeadModel)