import gc
import os
import warnings
from argparse import ArgumentParser
from datetime import datetime
from time import perf_counter, sleep

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
from transformers.utils import is_torch_sdpa_available

from config import settings

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def transcribe_audio(sample: str):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        settings.huggingface.whisper_model,
        torch_dtype=TORCH_TYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(settings.huggingface.whisper_model)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_TYPE,
        device=DEVICE,
        chunk_length_s=settings.huggingface.whisper_chunk_length_s,
        stride_length_s=settings.huggingface.whisper_stride_length_s,
        return_timestamps=True,
    )
    for chunk in pipe(sample).get("chunks", []):
        yield chunk

    del model, processor, pipe  # Clean up to free memory
    gc.collect()
    torch.cuda.empty_cache()


def create_resume(text: str) -> str:
    prompts = [
        {
            "role": "system",
            "content": "Você é um assistente de IA especializado em resumir transcrições, não invente fatos, mesmo que pareçam estar correlacionados, parecidos ou próximos com o que poderia ser a verdade. Se tem dúvida sobre algo ser verdadeiro ou falso, coloque uma marcação no fim da explicacao, algo como: [dúvida]. Tente ser mais técnico, pragmático e direto possível, não seja prolixo e / ou se repita demais. Esssas diretivas diretas que devem permear toda conversa.",
        },
        {
            "role": "user",
            "content": f"Faça um resumo, dos principais pontos chaves dessa transcrição de audio para texto. Retorne no seguinte do formato: - Resumo:\n <resumo médio de todo o conteúdo até 1024 palavras> \n\n- Pontos Chaves:\n <bullet point list de todo os principais pontos, pode adicionar nomes de pessoas se eu falar> ```{text}```",
        },
    ]

    model = AutoModelForCausalLM.from_pretrained(
        settings.huggingface.llm_model,
        torch_dtype=TORCH_TYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(settings.huggingface.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model.generate(
        inputs=tokenizer.apply_chat_template(prompts, add_generation_prompt=True, return_tensors="pt").to(DEVICE),
        streamer=streamer,
        max_new_tokens=settings.huggingface.llm_max_tokens,
        temperature=settings.huggingface.llm_temperature,
    )

    chunks = []
    for output in streamer:
        chunks.append(output)
        if len(chunks) == 10:
            yield "".join(chunks)
            chunks = []
    if chunks:
        yield "".join(chunks)

    del model, tokenizer, streamer  # Clean up to free memory
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


def main(sample: str, output_file: str):
    print(f"=> Transcribing audio file: {sample}")
    print(f"=> Output will be saved to: {output_file}")
    print(f"=> Torch Scale-Product-Attention (SDPA): {is_torch_sdpa_available()}")
    print(f"=> Using device: {DEVICE} with dtype: {TORCH_TYPE}")

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    print("=> Starting transcription...")
    start = perf_counter()
    all_transcribed_audio2_text = []
    for transcription in transcribe_audio(sample):
        transcribed_audio2_text = transcription.get("text", "")
        all_transcribed_audio2_text.append(transcribed_audio2_text)
        write_transcription(transcribed_audio2_text, f"audio/transcription-{date}.txt")
    print(f"=> Transcription completed in {(perf_counter() - start) // 60:.2f} minutes")
    sleep(5)
    print("=> Starting resume creation...")
    start = perf_counter()
    for resume in create_resume("".join(all_transcribed_audio2_text)):
        write_transcription(resume, output_file)
    print(f"=> Resume completed in {(perf_counter() - start) // 60:.2f} minutes")


if __name__ == "__main__":
    sample, output_file = parse_cli_args()
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"=> Removed existing file: {output_file}")
    main(sample, output_file)
