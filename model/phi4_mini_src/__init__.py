from transformers import AutoConfig, AutoImageProcessor, AutoFeatureExtractor, AutoModelForCausalLM
from .configuration_phi4mm import Phi4MMConfig
from .processing_phi4mm import Phi4MMProcessor, Phi4MMImageProcessor, Phi4MMAudioFeatureExtractor, speechlib_mel
from .modeling_phi4mm import Phi4MMForCausalLM


AutoImageProcessor.register("Phi4MMImageProcessor", Phi4MMImageProcessor)
AutoFeatureExtractor.register("Phi4MMAudioFeatureExtractor", Phi4MMAudioFeatureExtractor)
AutoConfig.register("phi4mm", Phi4MMConfig)
AutoModelForCausalLM.register(Phi4MMConfig, Phi4MMForCausalLM)
Phi4MMConfig.register_for_auto_class()
Phi4MMForCausalLM.register_for_auto_class("AutoModelForCausalLM")