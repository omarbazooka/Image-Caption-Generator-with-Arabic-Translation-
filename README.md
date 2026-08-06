# Image Caption Generator with Arabic Translation

A simple AI demo that generates an English caption for an uploaded image using **BLIP**, then translates the caption into Arabic using **MarianMT**.

## Live demo interface

The project now includes a clean Streamlit web interface designed for portfolio and live-demo use:

- Drag-and-drop image upload
- Image preview
- One-click caption generation
- English caption and Arabic translation shown side by side
- Cached model loading after the first run
- Clear loading and error states

## Models

- **Image captioning:** `Salesforce/blip-image-captioning-base`
- **English → Arabic translation:** `Helsinki-NLP/opus-mt-tc-big-en-ar`

## Run locally

```bash
git clone https://github.com/omarbazooka/Image-Caption-Generator-with-Arabic-Translation-.git
cd Image-Caption-Generator-with-Arabic-Translation-
python -m venv .venv
```

Activate the virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start the web app:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

> The first run can take longer because the pretrained Hugging Face models need to be downloaded. Later runs use the local cache.

## Project flow

```text
Uploaded Image
     ↓
BLIP Image Captioning
     ↓
English Caption
     ↓
MarianMT Translation
     ↓
Arabic Caption
```

## Legacy desktop app

The original Tkinter implementation is kept in:

```text
img-caption project/gui_caption_translate.py
```

## Tech stack

Python · PyTorch · Hugging Face Transformers · BLIP · MarianMT · Streamlit · Pillow
