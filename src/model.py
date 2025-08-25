import asyncio
import warnings
from typing import Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
from transformers import logging as transformers_logging

from src.config import settings

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = torch.float16 if torch.cuda.is_available() else torch.float32


class AudioModel:
    def __init__(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            settings.huggingface.whisper_model,
            torch_dtype=TORCH_TYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(DEVICE)

        self.processor = AutoProcessor.from_pretrained(settings.huggingface.whisper_model)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=TORCH_TYPE,
            device=DEVICE,
            chunk_length_s=settings.huggingface.whisper_chunk_length_s,
            stride_length_s=settings.huggingface.whisper_stride_length_s,
            return_timestamps=True,
        )

    async def execute(self, sample: str) -> Generator[str, None, None]:
        for chunk in self.pipe(sample).get("chunks", []):
            yield chunk
            await asyncio.sleep(0)


class TextModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.huggingface.llm_model,
            torch_dtype=TORCH_TYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        ).to(DEVICE)

        self.tokenizer = AutoTokenizer.from_pretrained(settings.huggingface.llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    async def execute(self, text: str) -> Generator[str, None, None]:
        prompts = [
            {
                "role": "system",
                "content": settings.system_prompt,
            },
            {
                "role": "user",
                "content": f"{settings.user_resume_prompt}\n\n```{text}```",
            },
        ]

        self.model.generate(
            inputs=self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(DEVICE),
            streamer=self.streamer,
            max_new_tokens=settings.huggingface.llm_max_tokens,
            temperature=settings.huggingface.llm_temperature,
        )

        chunks = []
        for output in self.streamer:
            chunks.append(output)
            if len(chunks) == 10:
                yield "".join(chunks)
                await asyncio.sleep(0)
                chunks = []
        if chunks:
            yield "".join(chunks)
            await asyncio.sleep(0)
