# app.py
"""
Streamlit UI for VisionCraft — AI Text-to-Image Generator (Talrn internship project).

Key points:
- Uses open-source Stable Diffusion models via Hugging Face diffusers.
- Local-first: does NOT force or require downloads at runtime.
- Fallback to SD v1.5 is handled in generate.py if needed.
"""

import io
import json
from pathlib import Path
import time as _time
from typing import List

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from generate import TextToImageGenerator
from utils import is_safe_prompt, apply_style_to_prompt

# -------------------------------------------------------------------------
# Config & output directory
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="🎨 VisionCraft — AI Text-to-Image Generator",
    layout="wide",
)
OUT_DIR = Path("generated_images")
OUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------------------
# Watermark helper
# -------------------------------------------------------------------------
def add_watermark(img: Image.Image, text: str = "AI Generated • VisionCraft"):
    """Add a small watermark to the bottom-right corner of the image."""
    watermarked = img.copy()
    draw = ImageDraw.Draw(watermarked)

    font_size = max(18, img.width // 40)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    text_w, text_h = draw.textsize(text, font=font)
    padding = 12
    position = (img.width - text_w - padding, img.height - text_h - padding)

    draw.text(position, text, font=font, fill=(255, 255, 255, 200))
    return watermarked


# -------------------------------------------------------------------------
# Sidebar: settings
# -------------------------------------------------------------------------
st.sidebar.title("Settings")

# Short model labels mapped to HF IDs
MODEL_OPTIONS = {
    "SD v1.5 (standard)": "runwayml/stable-diffusion-v1-5",
    "SD Turbo (fast)": "stabilityai/sd-turbo",
}

model_label = st.sidebar.selectbox(
    "Model",
    list(MODEL_OPTIONS.keys()),
    index=0,
)
model_choice = MODEL_OPTIONS[model_label]

st.sidebar.info(
    "Powered by Stable Diffusion.\n"
    "Runs locally. Images are AI-generated and watermarked."
)

num_images = st.sidebar.slider("Number of images", 1, 4, 1)
steps = st.sidebar.slider("Inference steps", 10, 50, 20)
guidance = st.sidebar.slider("Guidance scale (CFG)", 1.0, 12.0, 7.5)

fmt = st.sidebar.selectbox("Output format", ("PNG", "JPEG"))

image_size = st.sidebar.selectbox(
    "Image resolution",
    ("512x512", "768x512"),
    index=0,
)
width, height = map(int, image_size.split("x"))

st.sidebar.markdown("---")
st.sidebar.header("Style & Prompt Engineering")

style_choice = st.sidebar.selectbox(
    "Style",
    ("Default", "Photorealistic", "Artistic", "Cartoon", "Cyberpunk", "VanGogh"),
)

use_prompt_engineering = st.sidebar.checkbox(
    "Use prompt engineering (add quality tokens)",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.header("Saving / Filenames")

filename_prefix = st.sidebar.text_input(
    "Filename prefix (optional)",
    value="",
)

st.sidebar.markdown(
    "Images and metadata are saved in the `generated_images/` folder."
)


# -------------------------------------------------------------------------
# Main page content
# -------------------------------------------------------------------------
st.title("🎨 VisionCraft — AI Text-to-Image Generator")
st.caption("Remote Machine Learning Internship Project — Talrn (2025)")

st.markdown(
    """
VisionCraft converts your text prompts into AI-generated images using diffusion models.

**How to use:**
1. Enter a prompt and (optionally) a negative prompt.
2. Choose model, style, and settings from the sidebar.
3. Click **Generate Image(s)** and download the results.
    """
)

prompt = st.text_area(
    "Prompt",
    placeholder="a cyberpunk Indian street at night, neon lights",
    height=140,
)

negative_prompt = st.text_input(
    "Negative prompt (optional) — e.g., ugly, lowres, bad anatomy",
    value="",
)

col_main, col_ctrl = st.columns([3, 1])

with col_ctrl:
    st.markdown("### Controls")
    generate_btn = st.button("Generate Image(s)")
    st.caption("Generation time depends on your CPU/GPU and settings.")


# -------------------------------------------------------------------------
# Helper: save image & metadata
# -------------------------------------------------------------------------
def _save_image_and_meta(
    img: Image.Image,
    prompt_text: str,
    negative: str,
    params: dict,
    prefix: str,
    fmt_str: str,
) -> Path:
    """Save a single image and its metadata to OUT_DIR."""
    ts = int(_time.time())
    base_name = prefix.strip() or f"prompt_{ts}"

    # Ensure unique filename
    idx = 1
    while True:
        out_name = f"{base_name}_{idx}"
        out_path = OUT_DIR / f"{out_name}.{fmt_str.lower()}"
        meta_path = OUT_DIR / f"{out_name}.json"
        if not out_path.exists() and not meta_path.exists():
            break
        idx += 1

    # Save the image
    img.save(out_path, format=fmt_str)

    # Save metadata
    meta = {
        "prompt": prompt_text,
        "negative_prompt": negative,
        "model_requested": params.get("model_requested"),
        "model_used": params.get("model_used"),
        "num_images": params.get("num_images"),
        "steps": params.get("steps"),
        "guidance": params.get("guidance"),
        "style": params.get("style"),
        "timestamp": ts,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_path


# -------------------------------------------------------------------------
# Generate on button press
# -------------------------------------------------------------------------
if generate_btn:
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    elif not is_safe_prompt(prompt):
        st.error("❌ Unsafe / banned words detected. Please modify your prompt.")
    else:
        # Apply style / prompt engineering if enabled
        final_prompt = apply_style_to_prompt(prompt, style_choice) if use_prompt_engineering else prompt

        # Very rough estimate mainly for CPU
        est_secs = max(5, int(steps * num_images * 1.5))
        st.info(f"Estimated time: ~{est_secs} seconds for {num_images} image(s) (CPU).")

        with st.spinner("Generating images..."):
            try:
                gen = TextToImageGenerator()
                images: List[Image.Image] = gen.generate(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt or None,
                    num_images=num_images,
                    guidance_scale=guidance,
                    steps=steps,
                    width=width,
                    height=height,
                    model_id=model_choice,
                    allow_download=False,  # no downloads at runtime
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.exception(e)
                images = []

        if not images:
            st.warning("No images were returned by the generator.")
        else:
            st.success("Image(s) generated successfully!")

            for i, img_obj in enumerate(images):
                # Normalize to PIL.Image
                if isinstance(img_obj, (str, Path)):
                    try:
                        display_img = Image.open(img_obj)
                    except Exception:
                        display_img = img_obj
                else:
                    display_img = img_obj

                # Apply watermark
                display_img = add_watermark(display_img)

                # Show image
                st.image(
                    display_img,
                    caption=f"Image {i + 1}",
                    width="stretch",
                )

                # Save image & metadata
                params = {
                    "model_requested": model_choice,
                    "model_used": getattr(gen, "last_loaded_model", model_choice),
                    "num_images": num_images,
                    "steps": steps,
                    "guidance": guidance,
                    "style": style_choice,
                }
                out_path = _save_image_and_meta(
                    display_img,
                    final_prompt,
                    negative_prompt,
                    params,
                    filename_prefix,
                    fmt,
                )

                # Prepare download buffer
                buf = io.BytesIO()
                display_img.save(buf, format=fmt)
                buf.seek(0)

                st.download_button(
                    label=f"Download image {i + 1}",
                    data=buf,
                    file_name=out_path.name,
                    mime=f"image/{fmt.lower()}",
                )


# -------------------------------------------------------------------------
# Footer / notes
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "*All images are AI-generated using open-source diffusion models. "
    "Watermarks and basic prompt filtering are applied for responsible use.*"
)
