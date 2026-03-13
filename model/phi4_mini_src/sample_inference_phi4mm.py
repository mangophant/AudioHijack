import os
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

model_path = './'

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    _attn_implementation='flash_attention_2',
).cuda()
print("model.config._attn_implementation:", model.config._attn_implementation)

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
 
#################################################### text-only ####################################################
prompt = f'{user_prompt}what is the answer for 1+1? Explain it.{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')
inputs = processor(prompt, images=None, return_tensors='pt').to('cuda:0')

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f'>>> Response\n{response}')

#################################################### vision (single-turn) ####################################################
# single-image prompt
prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
print(f'>>> Prompt\n{prompt}')
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

#################################################### vision (multi-turn) ####################################################
# chat template
chat = [
    {'role': 'user', 'content': f'<|image_1|>What is shown in this image?'},
    {
        'role': 'assistant',
        'content': "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
    },
    {'role': 'user', 'content': 'What is so special about this image'},
]
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(prompt, [image], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

########################### vision (multi-frame) ################################
images = []
placeholder = ''
for i in range(1, 5):
    url = f'https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg'
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f'<|image_{i}|>'

messages = [
    {'role': 'user', 'content': placeholder + 'Summarize the deck of slides.'},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f'>>> Prompt\n{prompt}')

inputs = processor(prompt, images, return_tensors='pt').to('cuda:0')

generation_args = {
    'max_new_tokens': 1000,
    'temperature': 0.0,
    'do_sample': False,
}

generate_ids = model.generate(
    **inputs, **generation_args, generation_config=generation_config,
)

# remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)

# NOTE: Please prepare the audio file 'examples/what_is_the_traffic_sign_in_the_image.wav'
#       and audio file 'examples/what_is_shown_in_this_image.wav' before running the following code
#       Basically you can record your own voice for the question "What is the traffic sign in the image?" in "examples/what_is_the_traffic_sign_in_the_image.wav".
#       And you can record your own voice for the question "What is shown in this image?" in "examples/what_is_shown_in_this_image.wav".

AUDIO_FILE_1 = 'examples/what_is_the_traffic_sign_in_the_image.wav'
AUDIO_FILE_2 = 'examples/what_is_shown_in_this_image.wav'

if not os.path.exists(AUDIO_FILE_1):
    raise FileNotFoundError(f'Please prepare the audio file {AUDIO_FILE_1} before running the following code.')
########################## vision-speech ################################
prompt = f'{user_prompt}<|image_1|><|audio_1|>{prompt_suffix}{assistant_prompt}'
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
print(f'>>> Prompt\n{prompt}')
image = Image.open(requests.get(url, stream=True).raw)
audio = soundfile.read(AUDIO_FILE_1)
inputs = processor(text=prompt, images=[image], audios=[audio], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

########################## speech only ################################
speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

print(f'>>> Prompt\n{prompt}')
audio = soundfile.read(AUDIO_FILE_1)
inputs = processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

if not os.path.exists(AUDIO_FILE_2):
    raise FileNotFoundError(f'Please prepare the audio file {AUDIO_FILE_2} before running the following code.')
########################### speech only (multi-turn) ################################
audio_1 = soundfile.read(AUDIO_FILE_2)
audio_2 = soundfile.read(AUDIO_FILE_1)
chat = [
    {'role': 'user', 'content': f'<|audio_1|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.'},
    {
        'role': 'assistant',
        'content': "What is shown in this image.",
    },
    {'role': 'user', 'content': f'<|audio_2|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.'},
]
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(text=prompt, audios=[audio_1, audio_2], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

#################################################### vision-speech (multi-turn) ####################################################
# chat template
audio_1 = soundfile.read(AUDIO_FILE_2)
audio_2 = soundfile.read(AUDIO_FILE_1)
chat = [
    {'role': 'user', 'content': f'<|image_1|><|audio_1|>'},
    {
        'role': 'assistant',
        'content': "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
    },
    {'role': 'user', 'content': f'<|audio_2|>'},
]
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(text=prompt, images=[image], audios=[audio_1, audio_2], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
