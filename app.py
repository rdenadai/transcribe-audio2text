import os
import re
import uuid
from datetime import datetime
from time import perf_counter
from typing import Tuple

import markdown
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from main import create_resume, transcribe_audio

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Audio2Text + Resume")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Optional CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
OUTPUT_DIR = os.path.join(os.getcwd(), "audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_audio(file_path: str) -> Tuple[str, str, float, float]:
    # Transcribe
    t0 = perf_counter()
    transcript_chunks = []
    for chunk in transcribe_audio(file_path):
        transcript_chunks.append(chunk.get("text", ""))
    t1 = perf_counter()

    transcript_text = "".join(transcript_chunks)

    # Summarize
    summary_chunks = []
    for piece in create_resume(transcript_text):
        summary_chunks.append(piece)
    t2 = perf_counter()

    return transcript_text, "".join(summary_chunks), (t1 - t0), (t2 - t1)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
def upload_file(request: Request, file: UploadFile = File(...)):
    if file.content_type not in {"audio/mpeg", "audio/mp3"} and not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Envie um arquivo MP3 válido.")

    uid = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{uid}.mp3")

    # Persist the upload to disk
    with open(saved_path, "wb") as f:
        f.write(file.file.read())

    transcript_text = ""
    summary_text = ""
    t_transcribe = 0.0
    t_summarize = 0.0
    error = None

    try:
        # transcript_text, summary_text, t_transcribe, t_summarize = process_audio(saved_path)
        transcript_text = "Hey, how are you?"
        summary_text = """* **Resumo:**
  A transcrição contém apenas uma saudação inicial, onde a pessoa pergunta "Hey, how are you?". Não há desenvolvimento adicional de conteúdo, contexto ou troca de informações além da saudação.

* **Pontos Chaves:**

* Saudação: "Hey, how are you?"
"""
        t_transcribe = 0.0
        t_summarize = 0.0
        # Remove "think" part from summary if present (with start/end tags)
        summary_text = re.sub(r"<think>.*?</think>", "", summary_text, flags=re.IGNORECASE | re.DOTALL).strip()

        # Optional: persist outputs for audit/reference
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(os.path.join(OUTPUT_DIR, f"transcription-{ts}.txt"), "w", encoding="utf-8") as tf:
            tf.write(transcript_text)
        with open(os.path.join(OUTPUT_DIR, f"summary-{ts}.txt"), "w", encoding="utf-8") as sf:
            sf.write(summary_text)

    except Exception as e:
        error = str(e)
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except Exception:
            pass

    if error:
        return templates.TemplateResponse("index.html", {"request": request, "error": error})

    # Render transcript and summary as Markdown to HTML
    transcript_html = Markup(markdown.markdown(transcript_text, extensions=["extra"]))
    summary_html = Markup(markdown.markdown(summary_text, extensions=["extra"]))

    context = {
        "request": request,
        "transcript_html": transcript_html,
        "summary_html": summary_html,
        "t_transcribe": t_transcribe,
        "t_summarize": t_summarize,
    }
    return templates.TemplateResponse("index.html", context)
