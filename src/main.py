import asyncio
import gc
import os
import warnings
from argparse import ArgumentParser
from datetime import datetime
from time import perf_counter, sleep

import torch
from transformers import logging as transformers_logging
from transformers.utils import is_torch_sdpa_available

from src.model import AudioModel, TextModel

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = torch.float16 if torch.cuda.is_available() else torch.float32


async def transcribe_audio(sample: str):
    audio_model = AudioModel()
    async for chunk in audio_model.execute(sample):
        yield chunk
    del audio_model  # Clean up to free memory
    gc.collect()
    torch.cuda.empty_cache()


async def create_resume(text: str) -> str:
    text_model = TextModel()
    async for chunk in text_model.execute(text):
        yield chunk
    del text_model  # Clean up to free memory
    gc.collect()
    torch.cuda.empty_cache()


def write_transcription(text: str, output_file: str):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(text)


def parse_cli_args() -> tuple[str, str]:
    parser = ArgumentParser(description="Run Whisper ASR model on a sample audio file.")
    parser.add_argument(
        "--sample",
        type=str,
        default="data/sample.wav",
        help="Path to the sample audio file to transcribe.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="transcription.txt",
        help="Path to the output file where the transcription will be saved.",
    )
    args = parser.parse_args()
    sample = args.sample
    if not sample:
        raise ValueError("Please provide a valid audio file path.")
    output_file = args.output_file
    if not output_file:
        raise ValueError("Please provide a valid output file path.")
    return sample, output_file


async def main(sample: str, output_file: str):
    print(f"=> Transcribing audio file: {sample}")
    print(f"=> Output will be saved to: {output_file}")
    print(f"=> Torch Scale-Product-Attention (SDPA): {is_torch_sdpa_available()}")
    print(f"=> Using device: {DEVICE} with dtype: {TORCH_TYPE}")

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    print("=> Starting transcription...")
    start = perf_counter()
    all_transcribed_audio2_text = []
    async for transcription in transcribe_audio(sample):
        transcribed_audio2_text = transcription.get("text", "")
        all_transcribed_audio2_text.append(transcribed_audio2_text)
        write_transcription(transcribed_audio2_text, f"audio/transcription-{date}.txt")
    print(f"=> Transcription completed in {(perf_counter() - start) // 60:.2f} minutes")
    sleep(5)
    print("=> Starting resume creation...")
    start = perf_counter()
    async for resume in create_resume("".join(all_transcribed_audio2_text)):
        write_transcription(resume, output_file)
    print(f"=> Resume completed in {(perf_counter() - start) // 60:.2f} minutes")


if __name__ == "__main__":
    sample, output_file = parse_cli_args()
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"=> Removed existing file: {output_file}")
    asyncio.run(main(sample, output_file))
