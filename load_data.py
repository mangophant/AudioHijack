import os
import math
import json
import argparse
import torch
from datasets import load_from_disk, load_dataset, concatenate_datasets

import util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--num', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--audio_qa_path', type=str, default='data/AirBenchChat')
    parser.add_argument('--voice_chat_path', type=str, default='data/VoiceBench')
    args = parser.parse_args()
    
    print('loading dataset from disk or huggingface...')
    airbench = load_from_disk(args.audio_qa_path)
    airbench = airbench.filter(lambda x: util.check_audio_length(x, min_len=5, max_len=25))
    task_names = ["speech_QA", "sound_QA", "music_QA"]
    subsets = []
    for i, task in enumerate(task_names):
        task_subset = airbench.filter(lambda x: x["task_name"] == i)
        print(task, len(task_subset))
        task_subset = task_subset.shuffle(seed=args.seed).select(range(args.num))
        subsets.append(task_subset)
    audio_qa_dataset = concatenate_datasets(subsets)
    voicebench = load_dataset(
        path=args.voice_chat_path,
        data_dir="wildvoice", split="test"
    )
    voicebench = voicebench.filter(lambda x: util.check_audio_length(x, min_len=3, max_len=10))
    voice_chat_dataset = voicebench.shuffle(args.seed).select(range(args.num))
    
    print('preprocessing audios...')
    audio_dir = os.path.join(args.output_dir, 'benign')
    util.set_dir(audio_dir)
    data_infos = {'speech': [], 'sound': [], 'music': []}
    for data in audio_qa_dataset:
        text_prompt = data['question']
        audio_type = audio_qa_dataset.features['task_name'].int2str(data['task_name'])[:-3]
        sr = data['audio']['sampling_rate']
        num_audio_type = len(data_infos[audio_type])
        trial = audio_type + str(num_audio_type).zfill(3) + '.wav'
        audio = torch.from_numpy(data['audio']['array'])[:sr*20].float()
        audio = util.norm_audio(audio)
        audio_length = math.ceil(audio.shape[-1] / sr)
        util.save_audio(audio, os.path.join(audio_dir, trial), sr=sr)
        data_infos[audio_type].append({
            'trial': trial,
            'audio_type': audio_type,
            'audio_length': audio_length,
            'text_prompt': text_prompt,
            'speech_prompt': 'voice' + str(num_audio_type).zfill(3) + '.wav'
        })
    data_infos = [data_info for data_infos in list(data_infos.values()) for data_info in data_infos]
    for i, data in enumerate(voice_chat_dataset):
        # text_prompt = data['prompt']
        sr = data['audio']['sampling_rate']
        trial = 'voice' + str(i).zfill(3) + '.wav'
        audio = torch.from_numpy(data['audio']['array']).float()
        audio = util.norm_audio(audio)
        util.save_audio(audio, os.path.join(audio_dir, trial), sr=sr)
    data_path = os.path.join(args.output_dir, 'benign.jsonl')
    with open(data_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join([json.dumps(data_info, ensure_ascii=False) for data_info in data_infos]))
    print('benign dataset file saved at ', data_path)