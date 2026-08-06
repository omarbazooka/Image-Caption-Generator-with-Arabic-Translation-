# Image Caption Generator with Arabic Translation

A bilingual AI web application that generates an English description for an uploaded image using **BLIP**, then translates that caption into Arabic using **MarianMT**.

The project uses a lightweight custom web interface backed by **FastAPI**, so the live demo uses the real pretrained models rather than a mocked browser-only prediction.

## Features

- Drag-and-drop image upload
- Image preview before inference
- Real BLIP image caption generation
- English → Arabic translation with MarianMT
- English and Arabic result cards
- Copy-caption actions
- Responsive presentation-friendly frontend
- Clear loading, validation, and error states
- Models are loaded once and reused across requests
- Automatic CUDA usage when a compatible GPU is available

## Models

- **Image captioning:** `Salesforce/blip-image-captioning-base`
- **English → Arabic translation:** `Helsinki-NLP/opus-mt-tc-big-en-ar`

## Architecture

```text
Browser
  │
  │ image upload
  ▼
FastAPI /api/caption
  │
  ▼
BLIP Image Captioning
  │
  │ English caption
  ▼
MarianMT Translation
  │
  │ Arabic caption
  ▼
JSON response → Web UI
```

## Project structure

```text
.
├── app.py
├── requirements.txt
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── img-caption project/
    └── gui_caption_translate.py   # original Tkinter version
```

## Run locally

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/omarbazooka/Image-Caption-Generator-with-Arabic-Translation-.git
cd Image-Caption-Generator-with-Arabic-Translation-
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

For development with automatic reload:

```bash
uvicorn app:app --reload
```

> The first caption request can take longer because the pretrained Hugging Face models must be downloaded and loaded into memory. Later requests reuse the loaded models.

## API

### `POST /api/caption`

Send an image as multipart form data using the field name `image`.

Supported formats:

- JPEG
- PNG
- WebP

Maximum upload size: **8 MB**.

Example response:

```json
{
  "english_caption": "a dog running through a field",
  "arabic_caption": "كلب يركض عبر حقل",
  "models": {
    "captioning": "Salesforce/blip-image-captioning-base",
    "translation": "Helsinki-NLP/opus-mt-tc-big-en-ar"
  }
}
```

### `GET /api/health`

Returns the application status, selected models, and whether the AI models have already been loaded into memory.

## Tech stack

**AI / Backend**

- Python
- PyTorch
- Hugging Face Transformers
- BLIP
- MarianMT
- FastAPI

**Frontend**

- HTML5
- CSS3
- Vanilla JavaScript

## Original desktop version

The original Tkinter implementation is still available at:

```text
img-caption project/gui_caption_translate.py
```

It is kept as project history; the main application entry point is now the FastAPI web app in `app.py`.
