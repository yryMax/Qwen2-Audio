from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import yaml
import librosa
import os
import torch
import json

import time

start_time = time.time() # 记录开始时间


model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True).to("cuda:0").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)

# prompt = "<|audio_bos|><|AUDIO|><|audio_bos|>Translate this English speech into German:"
# audio, sr = librosa
# inputs = processor(text=prompt, audios=audio, return_tensors="pt")

wavs_dir = "/mnt/workspace/yangfei/datasets/IWSLT.OfflineTask/data/en-de/tst2022/wavs"
yamlPath = "/mnt/workspace/yangfei/datasets/IWSLT.OfflineTask/data/en-de/tst2022/IWSLT.TED.tst2022.en-de.yaml"
f = open(yamlPath,'r')
segmentsList = yaml.load(f,Loader=yaml.FullLoader)

last_end_time = start_time
full_audio_translations = {}
audio, sr = None, None
history = []
current_file_translation_segments = []
last_file_name = "dummy"
for segment in segmentsList:
    filename = segment['wav'].split('/')[-1]
    wav_path = os.path.join(wavs_dir,filename)
    offset = segment['offset']
    duration = segment['duration']
    if filename != last_file_name:
        history=[]
        if last_file_name!='dummy':
            full_audio_translations[last_file_name]="||".join(current_file_translation_segments)
            curr_end_time = time.time()
            elapsed_time = curr_end_time - last_end_time
            print(f"本文件转录所用时间： {elapsed_time}秒")
            last_end_time = curr_end_time
        last_file_name = filename
        current_file_translation_segments=[]
        audio, sr = librosa.load(wav_path,sr=16000)
    audio_clip = audio[int(offset*sr):int((offset+duration)*sr)]
    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and translate the speech into German: <|en|>"
    inputs = processor(text=prompt,audio=audio_clip,sampling_rate=sr,return_tensors='pt')
    # 将输入移动到GPU
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")

    # # --- DEBUG: 打印 inputs 的内容 ---
    # print("\n--- Debugging inputs for model.generate ---")
    # for k, v in inputs.items():
    #     print(f"Key: {k}, Type: {type(v)}, Shape (if Tensor): {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")
    #     if not isinstance(v, torch.Tensor):
    #         print(f"  WARNING: Value for key '{k}' is NOT a torch.Tensor. It's a {type(v)}.")
    # print("--- End Debugging ---")
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    current_file_translation_segments.append(response.strip())

if last_file_name!='dummy':
    full_audio_translations[last_file_name]="||".join(current_file_translation_segments)

output_dir = "./output_translations" # 定义输出目录
os.makedirs(output_dir, exist_ok=True) # 创建输出目录如果不存在


output_filename = os.path.join(output_dir, f"qwen2_audio_translations_en_de_tst2022.json")

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(full_audio_translations, f, ensure_ascii=False, indent=4)

print(f"All translations saved to: {output_filename}")
print("\nProcessing complete.")


end_time = time.time()   # 记录结束时间
elapsed_time = end_time - last_end_time
print(f"本文件转录所用时间： {elapsed_time}秒")

total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
remaining_seconds = total_seconds % 3600
minutes = int(remaining_seconds // 60)
seconds = remaining_seconds % 60

if hours > 0:
    print(f"\n脚本总执行时间: {hours} 小时 {minutes} 分 {seconds:.2f} 秒")
elif minutes > 0:
    print(f"\n脚本总执行时间: {minutes} 分 {seconds:.2f} 秒")
else:
    print(f"\n脚本总执行时间: {seconds:.2f} 秒")