import argparse
import json
import time
from itertools import islice
import numpy as np
import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from chunk2 import ChunkedAudioGenerator
import librosa
import transformers
transformers.logging.set_verbosity_error()
ds_collections = {
    'long_translation': {'path': "/mnt/workspace/renyi/datasets/LongSpeechQA/translationQA.jsonl"}
}

lang_name = {
  "en": "English",
  "de": "German",
  "fr": "French",
  "zh-CN": "Mandarin",
  "es": "Spanish",
  "it": "Italian",
  "pl": "Polish",
  "ro": "Romanian",
  "ja": "Japanese",
  "hu": "Hungarian",
  "cs": "Czech",
  "tr": "Turkish",
  "nl": "Dutch",
  "th": "Thai",
  "id": "Indonesian",
  "vi": "Vietnamese",
  "ko": "Korean"
}


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, ds, amount=2004):
        path = ds['path']
        with open(path) as file:
            self.datas = list(islice(file, amount))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['messages'][0]['audio']
        target_lang = data['language']
        message = f"Detect the language and translate the speech into {lang_name[target_lang]}: <|en|>"
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + message
        gt = data['messages'][1]['content']
        instr = message

        return {
            'audio': audio,
            'prompt': prompt,
            'source': target_lang,
            'gt': gt,
            'instr': prompt
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='long_translation')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cache_dir = "/mnt/workspace/renyi/Qwen2-Audio"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, 
        device_map='cuda', 
        trust_remote_code=True, 
        torch_dtype='auto',
        cache_dir=cache_dir
    ).eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint, cache_dir=cache_dir)
    processor.tokenizer.padding_side = 'left'

    chunk_g = ChunkedAudioGenerator(model, processor)

    dataset = AudioDataset(ds=ds_collections[args.dataset])
    
    gts = []
    sources = []
    rets = []
    audio_paths = []
    instrs = []
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        
        audio, _ = librosa.load(sample['audio'], sr=processor.feature_extractor.sampling_rate)
        
        output = chunk_g.generate_with_chunked_audio(
            text_prompt=sample['instr'], 
            audio=audio
        )
        
        gts.append(sample['gt'])
        rets.append(output)
        sources.append(sample['source'])
        audio_paths.append(sample['audio'])
        instrs.append(sample['instr'])
        

    print(f"\nEvaluating {args.dataset} ...")
    
    results = []
    for gt, response, source, audio_path, instr in zip(gts, rets, sources, audio_paths, instrs):
        results.append({
            'gt': gt,
            'response': response,
            'source': source,
            'audio_path': audio_path,
            'instr': instr,
        })
    
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_{time_prefix}.json'
    json.dump(results, open(results_file, 'w'))
    
    results_dict = {}
    for item in results:
        source = item["source"]
        results_dict.setdefault(source, []).append(item)

    bleu_all = []
    for source in results_dict:
        text_lan = source
        if text_lan == "ja":
            text_lan = "ja-mecab"
        elif text_lan == "zh-CN":
            text_lan = "zh"
        else:
            text_lan = "13a"
        
        refs, hyps = [], []
        results_list = results_dict[source]
        for result in results_list:
            gt = result["gt"]
            response = result["response"]
            refs.append(gt)
            hyps.append(response)


        with open('output.txt', 'a', encoding='utf-8') as file:
            file.write(f'Source: {source}\n')
            for result in results_list:
                gt = result["gt"]
                response = result["response"]
                audio_link = result.get("audio_path", "N/A")
                prompt = result.get("instr", "N/A")

                file.write(f'GT: {gt}\n')
                file.write(f'HYP: {response}\n')
                file.write(f'Audio Link: {audio_link}\n')
                file.write(f'Prompt: {prompt}\n')
                file.write('---\n')

        if refs: 
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=text_lan).score
            bleu_all.append(bleu)
            print(f"source: {source}  cnt: {len(refs)} bleu score: {bleu:.4f}")
    
    if bleu_all:
        print(f"\nOverall BLEU - Mean: {np.mean(bleu_all):.4f}, Std: {np.std(bleu_all):.4f}, "
              f"Max: {np.max(bleu_all):.4f}, Min: {np.min(bleu_all):.4f}")