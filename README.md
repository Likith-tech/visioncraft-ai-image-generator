
# VisionCraft — AI Text-to-Image Generator

VisionCraft is a simple, local-first **AI text-to-image generator** built as part of the **Remote Machine Learning Internship Task at Talrn (2025)**.

It uses open-source **Stable Diffusion** models to convert text prompts into images, with support for multiple styles, negative prompts, adjustable generation settings, and automatic watermarking for responsible AI use.

---

## ✨ Features

- **Text-to-image generation** using open-source Stable Diffusion models
- **Multiple styles**: Default, Photorealistic, Artistic, Cartoon, Cyberpunk, Van Gogh
- **Negative prompts** to reduce unwanted artifacts (e.g., *ugly, lowres, bad anatomy*)
- Adjustable:
  - Number of images per prompt (batch size)
  - Inference steps
  - Guidance scale (CFG)
  - Output resolution (512×512, 768×512)
- **Simple web UI** built with Streamlit
- **Image downloads** in PNG or JPEG
- **Metadata saving** (prompt, timestamp, model, parameters)
- **Basic safety filter** for prompts (keyword-based)
- **Automatic AI watermark** on all outputs:  
  _“AI Generated • VisionCraft”_

---

## 🏗️ Architecture Overview

### High-level components

1. **Frontend (UI)** – `app.py`
   - Built with **Streamlit**
   - Allows the user to:
     - Enter prompt and negative prompt
     - Choose model, style, and generation settings
     - View generated images
     - Download images

2. **Generation Engine** – `generate.py`
   - Uses **Hugging Face diffusers** + **PyTorch**
   - Loads Stable Diffusion pipelines:
     - `runwayml/stable-diffusion-v1-5` (standard)
     - `stabilityai/sd-turbo` (fast)
   - Auto-selects device: **GPU (`cuda`) if available, else CPU**
   - Local-first:
     - Tries to load models with `local_files_only=True`
     - If the selected model is not available locally and downloads are disabled, it attempts to fall back to `runwayml/stable-diffusion-v1-5`

3. **Utilities** – `utils.py`
   - `is_safe_prompt(prompt)` – simple keyword-based safety filter
   - `apply_style_to_prompt(prompt, style)` – appends style-specific quality tokens to the prompt

4. **Storage**
   - All generated images and metadata are stored under:
     - `generated_images/`
   - Each image has a corresponding `.json` metadata file.

---

## 🧪 Models Used

VisionCraft supports the following open-source text-to-image models via Hugging Face:

- **SD v1.5 (standard)** → `runwayml/stable-diffusion-v1-5`
- **SD Turbo (fast)** → `stabilityai/sd-turbo`

> Note: The repository does **not** include model weights.  
> Models must be available in the local Hugging Face cache or downloaded once on the host machine (outside this repo).

---

## ⚙️ Tech Stack

- **Language**: Python 3.10+ (tested on 3.11)
- **Frameworks / Libraries**:
  - [Streamlit] – web UI
  - [PyTorch] – tensor computations, model execution
  - [diffusers] – Stable Diffusion pipelines
  - [Pillow (PIL)] – image handling & watermarking
- **Other**:
  - `logging`, `pathlib`, `json`, `time`

---

## 💻 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```
###2. Create & activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

###3. Install dependencies
pip install -r requirements.txt

🚀 Running the Application
streamlit run app.py


Open the local server link (usually http://localhost:8501/)
## 🕹 How to Use

1. Enter a **prompt** describing the image you want to generate
2. *(Optional)* Enter a **negative prompt**
3. Adjust settings in the sidebar:
   - Number of images
   - Resolution
   - Inference steps
   - Guidance scale
   - Style
   - Output format
4. Choose the model (`SD v1.5` or `SD Turbo`)
5. Click **Generate Image(s)**
6. View the generated results
7. Click **Download image** to save outputs

All generated images & metadata are automatically stored in:
generated_images/

## 🎨 Prompt Engineering Tips

### Example prompts
a futuristic cyberpunk street at night, neon reflections, ultra realistic, 8k
a cute baby dragon reading a book, soft lighting, watercolor illustration
a majestic Indian temple at sunrise, golden light, high detail, DSLR lens

### Negative prompt example
ugly, extra limbs, deformed, blurry, watermark, lowres


### Quality tokens
HDR, cinematic lighting, hyperdetailed, volumetric light, high contrast 


---

## 🧯 Ethical & Responsible AI Usage

- Unsafe keywords filtered
- Watermark added:
AI Generated • VisionCraft
- Offline / privacy-safe execution

---

## ⚡ CPU vs GPU Performance

| Hardware | Performance |
|-----------|-------------|
CPU | Slower (20–200 sec per image)
GPU (`cuda`) | Fast (2–10 sec per image)

Automatically selected via:


---

## 📁 Output & Metadata

### Output structure
generated_images/output_1.png
generated_images/output_1.json


### Example metadata
```json
{
  "prompt": "...",
  "model_used": "runwayml/stable-diffusion-v1-5",
  "guidance": 7.5,
  "steps": 20,
  "style": "Photorealistic",
  "timestamp": 123456
}
```
## ⚠ Limitations

- Requires local model cache
- Basic safety filter only
- Slower on CPU
- No fine-tuning / training support yet

---

## 🚀 Future Improvements

- SDXL, Turbo XL support
- ControlNet & LoRA support
- Image gallery / history page
- Fine-tuning & custom dataset training
- Advanced content safety filtering

---

## 📎 Sample Outputs

Upload example images under:
intern@talrn.com


---

## 👤 Author

**Vadranapu Likith**  
📍 India  
🎓 GITAM University (2023–2027)  
💻 AI/ML | Generative AI | Computer Vision

---

## ⭐ Support

If you like this project, please ⭐ star the repository 🙂

```bash
git push --set-upstream origin main


