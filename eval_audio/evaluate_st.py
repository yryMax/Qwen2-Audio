import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import sacrebleu
import torch
import requests
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from itertools import islice

ds_collections = {
    'long_translation': {'path': "/mnt/workspace/renyi/datasets/LongSpeechQA/translationQA.jsonl"}
}


def json_from_this_to_that(source):
    ans = random.sample(lang_instr[source['target_lang']], 1)

    return {
        "language": source['target_lang'],
        "task": "translation",
        "messages": [
            {
                "role": "user",
                "audio": source['wav_path'],
                "content": ans
            },
            {
                "role": "assistant",
                "content": source['content']
            }
        ]

    }


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, ds, amount=50):
        path = ds['path']

        with open(path) as file:
            self.datas = list(islice(file, amount))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['messages'][0]['audio']
        target_lang = data['language']
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + data['messages'][0]['content']
        gt = data['messages'][1]['content']
        instr = data['messages'][0]['content']

        return {
            'audio': audio,
            'prompt': prompt,
            'source': target_lang,
            'gt': gt,
            'instr': instr
        }


def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(inputs, processor):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    instr = [_['instr'] for _ in inputs]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _
                    in inputs]
    inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate,
                       return_tensors="pt", padding=True)
    return inputs, audio_path, source, gt, instr


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='long_translation')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)

    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    gts = []
    sources = []
    rets = []
    audio_paths = []
    instrs = []
    for _, (inputs, audio_path, source, gt, instr) in tqdm(enumerate(data_loader)):
        inputs = {k: v.to('cuda') for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
        output_ids = output_ids[:, inputs['input_ids'].size(1):]
        output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        gts.extend(gt)
        rets.extend(output)
        sources.extend(source)
        audio_paths.extend(audio_path)
        instrs.extend(instr)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_gts = [None for _ in range(world_size)]
    merged_sources = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_audio_paths = [None for _ in range(world_size)]
    merged_instrs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_gts, gts)
    torch.distributed.all_gather_object(merged_sources, sources)
    torch.distributed.all_gather_object(merged_responses, rets)
    torch.distributed.all_gather_object(merged_audio_paths, audio_paths)
    torch.distributed.all_gather_object(merged_instrs, instrs)

    merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    merged_audio_paths = [_ for _ in itertools.chain.from_iterable(merged_audio_paths)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]
    merged_instrs = [_ for _ in itertools.chain.from_iterable(merged_instrs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for gt, response, source, audio_path, instr in zip(merged_gts, merged_responses, merged_sources, merged_audio_paths, merged_instrs):
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
        for item in tqdm(results):
            source = item["source"]
            results_dict.setdefault(source, []).append(item)
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
                # 写入 source 信息
                file.write(f'Source: {source}\n')

                # 遍历 results_list 并写入 gt, hyps, audio link 和 prompt
                for result in results_list:
                    gt = result["gt"]
                    response = result["response"]
                    audio_link = result.get("audio_path", "N/A")
                    prompt = result.get("instr", "N/A")

                    file.write(f'GT: {gt}\n')
                    file.write(f'HYP: {response}\n')
                    file.write(f'Audio Link: {audio_link}\n')
                    file.write(f'Prompt: {prompt}\n')
                    file.write('---\n')  # 分隔符

            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=text_lan).score
            print(f"source: {source}  cnt: {len(refs)} bleu score: {bleu:.4f}")

    torch.distributed.barrier()
