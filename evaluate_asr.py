import argparse
import json
import time
from itertools import islice
import re
import editdistance as ed
import torch
from pygments.lexers.sql import language_callback
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from chunk2 import ChunkedAudioGenerator
import librosa
import numpy as np
from evaluate_tokenizer import EvaluationTokenizer
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from cn_tn import TextNorm

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode='',
)
basic_normalizer = BasicTextNormalizer()

PUNCS = '!,.?;:'

ds_collections = {
    'long_asr': {'path': '/mnt/workspace/renyi/datasets/LongSpeechQA/ASRQA.jsonl', 'language': 'en'},
}


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, ds, amount=None):
        path = ds['path']
        with open(path) as file:
            if amount:
                self.datas = list(islice(file, amount))
            else:
                self.datas = list(file)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['messages'][0]['audio']
        gt = data['messages'][1]['content']

        language = data.get('language', 'en')

        prompt = data['messages'][0]['content']

        return {
            'audio': audio,
            'prompt': prompt,
            'gt': gt,
            'language': language,
        }


def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language.contains("zh"):
        gt = re.sub(rf"\s+", r"", gt)
    return gt


def compute_wer(refs, hyps, languages):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )

    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]

        if languages[i] == "en":
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        elif languages[i].contains("zh"):
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)

        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()

        if languages[i].contains("zh"):
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]


        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)

    return distance / ref_length if ref_length > 0 else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='long_asr')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amount', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--chunk-duration', type=int, default=30, help='Chunk duration in seconds')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cache_dir = "/mnt/workspace/renyi/Qwen2-Audio"

    print("Loading model...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint,
        device_map='cuda',
        torch_dtype='auto',
        trust_remote_code=True,
        cache_dir=cache_dir
    ).eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint, cache_dir=cache_dir)
    processor.tokenizer.padding_side = 'left'

    chunk_g = ChunkedAudioGenerator(model, processor, chunk_duration=args.chunk_duration)

    torch.manual_seed(args.seed)

    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
        amount=args.amount
    )

    gts = []
    rets = []
    sources = []
    audio_paths = []

    print("\nProcessing samples...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]

        try:
            audio, _ = librosa.load(sample['audio'], sr=processor.feature_extractor.sampling_rate)

            output = chunk_g.generate_with_chunked_audio(
                text_prompt=sample['prompt'],
                audio=audio,
                max_new_tokens=256,
                min_new_tokens=1,
                do_sample=False
            )

            gts.append(sample['gt'])
            rets.append(output)
            sources.append(sample['language'])
            audio_paths.append(sample['audio'])


        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")


    results = []
    for gt, response, source, audio_path in zip(gts, rets, sources, audio_paths):
        results.append({
            'gt': gt,
            'response': response,
            'source': source,
            'audio_path': audio_path,
        })

    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_{time_prefix}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


    all_wers = []
    refs = []
    hyps = []
    languages = [item["language"] for item in results]
    for item in results:
        gt = item["gt"]
        response = item["response"]

        gt = remove_sp(gt, item["language"])
        response = remove_sp(response, item["language"])

        refs.append(gt)
        hyps.append(response)

    wer = compute_wer(refs, hyps, languages)
    all_wers.append(wer)


    if all_wers:
        avg_wer = sum(all_wers) / len(all_wers)
        print(f"\n{'=' * 50}")
        print(f"Overall Statistics:")
        print(f"Average WER: {avg_wer:.4f}")

        output_file = f'asr_output.txt'
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f"ASR Evaluation Results\n")
            file.write(f"Average WER: {avg_wer:.4f} ({avg_wer * 100:.2f}%)\n")

            for i, result in enumerate(results):
                file.write(f"Sample {i + 1}:\n")
                file.write(f"Audio: {result['audio_path']}\n")
                file.write(f"GT: {result['gt'][:200]}...\n")
                file.write(f"HYP: {result['response'][:200]}...\n")
                file.write(f"{'-' * 80}\n")

    else:
        print("\nNo valid results to evaluate!")