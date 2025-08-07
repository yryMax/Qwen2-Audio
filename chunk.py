import torch
import librosa
import numpy as np
from typing import List, Optional, Union


class ChunkedAudioGenerator:
    """
    A wrapper class for generating responses from long audio inputs.
    Slice the audio and reuse the prompt, go through the processor and model.
    Regroup the output of each chunk to generate the final response.
    """
    def __init__(self, model, processor, chunk_duration=30):
        self.model = model
        self.processor = processor
        self.chunk_duration = chunk_duration
        self.sampling_rate = processor.feature_extractor.sampling_rate

    def chunk_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        chunk_samples = int(self.chunk_duration * self.sampling_rate)

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            chunks.append(chunk)

            if end >= len(audio):
                break
            start += chunk_samples

        return chunks

    def generate_with_chunked_audio(
            self,
            text_prompt: str,
            audio: np.ndarray,
            **generation_kwargs
    ) -> str:
        """
        Process a long audio input by splitting it into chunks,
        Args:
            text_prompt: 文本提示，包含<|AUDIO|>占位符
            audio: 长音频数组
            max_new_tokens: 每个chunk的最大生成token数
            **generation_kwargs: 其他生成参数
        """
        audio_chunks = self.chunk_audio(audio)
        print(f"Audio split into {len(audio_chunks)} chunks")

        chunk_responses = []
        accumulated_context = ""

        for i, chunk in enumerate(audio_chunks):
            print(f"Processing chunk {i + 1}/{len(audio_chunks)}")

            """
            TODO: new prompt based on the accumulated context.
            """

            inputs = self.processor(
                text=text_prompt,
                audios=chunk,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    **generation_kwargs
                )

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][input_length:]
            chunk_response = self.processor.decode(
                generated_ids,
                skip_special_tokens=True
            )

            chunk_responses.append(chunk_response)

            """
            TODO: update context with the new response.
            """

        final_response = self._merge_responses(chunk_responses)
        return final_response

    def _build_context_prompt(self, original_prompt: str, context: str, chunk_index: int) -> str:
        """为后续chunk构建包含上下文的prompt"""
        # 这里可以根据具体需求调整prompt格式
        context_prompt = f"""
        Previous audio analysis: {context}
        
        Continue analyzing the next part of the audio:
        {original_prompt}
        """
        return context_prompt

    def _update_context(self, current_context: str, new_response: str) -> str:
        """更新累积的上下文，避免上下文过长"""
        # 简单策略：保留最近的N个字符
        max_context_length = 1000
        combined = current_context + " " + new_response
        if len(combined) > max_context_length:
            combined = combined[-max_context_length:]
        return combined.strip()

    def _merge_responses(self, responses: List[str]) -> str:
        """合并多个chunk的响应"""
        return " ".join(responses)
