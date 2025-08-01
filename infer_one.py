from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor
from myqwen2audio import Qwen2AudioForConditionalGeneration



if __name__ == '__main__':
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and translate the speech into Mandarin: <|en|>"
    url = "/mnt/workspace/renyi/datasets/LongSpeech/wavs/000006.wav"
    audio, sr = librosa.load(url, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=prompt, audios=audio, return_tensors="pt")

    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response