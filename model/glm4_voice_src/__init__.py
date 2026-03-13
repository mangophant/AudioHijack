from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from .configuration_chatglm import ChatGLMConfig
from .tokenization_chatglm import ChatGLM4Tokenizer
from .modeling_chatglm import ChatGLMForConditionalGeneration
from .speech_tokenizer.modeling_whisper import WhisperVQEncoder


AutoConfig.register("chatglm", ChatGLMConfig)
AutoTokenizer.register(ChatGLMConfig, ChatGLM4Tokenizer)
AutoModel.register(ChatGLMConfig, ChatGLMForConditionalGeneration)
AutoModelForCausalLM.register(ChatGLMConfig, ChatGLMForConditionalGeneration)