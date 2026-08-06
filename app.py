from __future__ import annotations

import io
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

app = FastAPI(
    title="Image Caption Generator API",
    description="Generate an English image caption with BLIP and translate it to Arabic with MarianMT.",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_models = None
_model_lock = threading.Lock()


def load_models():
    """Load the pretrained models once and reuse them for later requests."""
    global _models

    if _models is None:
        with _model_lock:
            if _models is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
                caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
                translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
                translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)

                caption_model.to(device).eval()
                translation_model.to(device).eval()

                _models = (
                    caption_processor,
                    caption_model,
                    translation_tokenizer,
                    translation_model,
                    device,
                )

    return _models


def generate_caption(image: Image.Image, processor, model, device: torch.device) -> str:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=40)

    return processor.decode(output[0], skip_special_tokens=True).strip()


def translate_to_arabic(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> str:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        translated = model.generate(**inputs, max_new_tokens=80)

    return tokenizer.decode(translated[0], skip_special_tokens=True).strip()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _models is not None,
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

    try:
        processor, caption_model, tokenizer, translation_model, device = load_models()
        english_caption = generate_caption(source_image, processor, caption_model, device)
        arabic_caption = translate_to_arabic(
            english_caption,
            tokenizer,
            translation_model,
            device,
        )
    except Exception as exc:
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
