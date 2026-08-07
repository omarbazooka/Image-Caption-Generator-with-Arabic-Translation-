from __future__ import annotations

import gc
import io
import logging
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    MarianMTModel,
    MarianTokenizer,
)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-ar"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LOGGER = logging.getLogger("image-caption-api")

app = FastAPI(
    title="Image Caption Generator API",
    description="Generate an English image caption with BLIP and translate it to Arabic with MarianMT.",
    version="2.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Running both transformer models at the same time creates a large memory spike.
# Serialize inference and load one model at a time so paid CPU instances can use
# the smallest practical amount of RAM.
_inference_lock = threading.Lock()


def release_memory(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_caption(image: Image.Image, device: torch.device) -> str:
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
    model.to(device).eval()

    try:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=40)

        return processor.decode(output[0], skip_special_tokens=True).strip()
    finally:
        release_memory(model, processor)


def translate_to_arabic(text: str, device: torch.device) -> str:
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)
    model.to(device).eval()

    try:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            translated = model.generate(**inputs, max_new_tokens=80)

        return tokenizer.decode(translated[0], skip_special_tokens=True).strip()
    finally:
        release_memory(model, tokenizer)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "loading_strategy": "sequential",
        "caption_model": CAPTION_MODEL,
        "translation_model": TRANSLATION_MODEL,
    }


@app.post("/api/caption")
async def caption_image(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Please upload JPG, PNG, or WebP.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller.")

    try:
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        with _inference_lock:
            english_caption = generate_caption(source_image, device)
            gc.collect()
            arabic_caption = translate_to_arabic(english_caption, device)
            gc.collect()
    except Exception as exc:
        LOGGER.exception("Image caption inference failed")
        raise HTTPException(
            status_code=500,
            detail="The AI models could not process this image. Please try again.",
        ) from exc

    return {
        "english_caption": english_caption,
        "arabic_caption": arabic_caption,
        "models": {
            "captioning": CAPTION_MODEL,
            "translation": TRANSLATION_MODEL,
        },
    }
