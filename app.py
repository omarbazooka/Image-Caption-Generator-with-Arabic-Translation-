import io

import streamlit as st
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    MarianMTModel,
    MarianTokenizer,
)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-ar"

st.set_page_config(
    page_title="Image Caption AI",
    page_icon="🖼️",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        .hero {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            margin-bottom: 0.35rem;
        }
        .hero p {
            color: #6b7280;
            font-size: 1.05rem;
        }
        .result-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 150px;
            background: rgba(128, 128, 128, 0.05);
        }
        .result-label {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.7;
            margin-bottom: 0.55rem;
        }
        .result-text {
            font-size: 1.05rem;
            line-height: 1.7;
        }
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-size: 1.12rem;
        }
        .model-badges {
            text-align: center;
            color: #6b7280;
            font-size: 0.85rem;
            margin-top: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models():
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
    translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)

    caption_model.eval()
    translation_model.eval()

    return (
        caption_processor,
        caption_model,
        translation_tokenizer,
        translation_model,
    )


def generate_caption(image: Image.Image, processor, model) -> str:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def translate_to_arabic(text: str, tokenizer, model) -> str:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        translated = model.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(translated[0], skip_special_tokens=True).strip()


st.markdown(
    """
    <div class="hero">
        <h1>🖼️ Image Caption AI</h1>
        <p>Upload an image and let AI describe it in English, then translate the caption into Arabic.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, PNG and WebP.",
)

if uploaded_file is None:
    st.info("Upload an image to start the demo.", icon="👆")
else:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.image(image, caption=uploaded_file.name, use_container_width=True)

    if st.button("✨ Generate Caption", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading AI models and analyzing the image..."):
                processor, caption_model, tokenizer, translation_model = load_models()
                caption = generate_caption(image, processor, caption_model)
                arabic_caption = translate_to_arabic(
                    caption, tokenizer, translation_model
                )

            st.success("Caption generated successfully.")
            left, right = st.columns(2, gap="medium")

            with left:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">English Caption</div>
                        <div class="result-text">{caption}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Arabic Translation</div>
                        <div class="result-text arabic-text">{arabic_caption}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(
                "The demo could not process this image. "
                "Check your internet connection on the first run and try again."
            )
            with st.expander("Technical details"):
                st.code(str(exc))

with st.expander("How it works"):
    st.markdown(
        """
        1. **BLIP** analyzes the uploaded image and generates an English caption.
        2. **MarianMT** translates the generated caption from English to Arabic.
        3. Both pretrained models are downloaded from Hugging Face on the first run and cached afterwards.
        """
    )

st.markdown(
    f"""
    <div class="model-badges">
        Captioning: <b>{CAPTION_MODEL}</b> &nbsp;•&nbsp; Translation: <b>{TRANSLATION_MODEL}</b>
    </div>
    """,
    unsafe_allow_html=True,
)
