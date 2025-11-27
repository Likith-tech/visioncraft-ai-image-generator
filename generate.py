# generate.py
"""
Text-to-image generator wrapper using Hugging Face diffusers.

Features:
- Local-first loading: tries to load the requested model with local_files_only=True.
- Fallback model: if requested model is not available locally and downloads are disabled,
  it falls back to runwayml/stable-diffusion-v1-5 (also local-only).
- Optional downloads: if allow_download=True, will attempt to download models from the Hub.
- Returns a list of PIL.Image.Image objects.
"""

import logging
from typing import List, Optional

from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextToImageGenerator:
    def __init__(self, device: Optional[str] = None):
        """
        device: Optional override, e.g., "cuda" or "cpu".
                If None, will auto-detect and use "cuda" if available, else "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # dtype selection: float16 on GPU if available, otherwise float32
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Remember the last successfully loaded model (for metadata/UI)
        self.last_loaded_model: Optional[str] = None

        # Cache of loaded pipelines to avoid reloading within a single process
        self._pipelines = {}

    # -------------------------------------------------------------------------
    # Internal: load pipeline
    # -------------------------------------------------------------------------
    def _load_pipeline(self, model_id: str, allow_download: bool = False) -> StableDiffusionPipeline:
        """
        Loads a StableDiffusionPipeline for model_id.

        Behavior:
          - Try local-only load first (local_files_only=True).
          - If local load succeeds -> return pipeline.
          - If local load fails:
              - If allow_download == False:
                  -> try fallback local-only model (runwayml/stable-diffusion-v1-5).
              - If allow_download == True:
                  -> attempt download for the requested model.
                  -> if that fails, attempt download for the fallback model.

        Raises:
          RuntimeError with a clear message if neither the requested nor fallback
          models can be loaded.
        """
        # If already loaded in this process, reuse from cache
        if model_id in self._pipelines:
            logger.info(f"Using cached pipeline for `{model_id}`.")
            return self._pipelines[model_id]

        fallback_id = "runwayml/stable-diffusion-v1-5"

        # 1) Try local-only load for requested model
        try:
            logger.info(f"Attempting to load model `{model_id}` from local cache (no downloads)...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                local_files_only=True,
            )
            logger.info(f"Loaded `{model_id}` from local cache.")
        except Exception as e_local:
            logger.warning(f"Local-only load failed for `{model_id}`: {e_local!r}")

            if not allow_download:
                # 2) Downloads are disabled -> try fallback local-only model
                logger.info(f"Downloads disabled. Trying fallback local model `{fallback_id}`...")
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        fallback_id,
                        torch_dtype=self.torch_dtype,
                        local_files_only=True,
                    )
                    logger.info(f"Loaded fallback model `{fallback_id}` from local cache.")
                    model_id = fallback_id  # for metadata
                except Exception as e_fb:
                    logger.error(f"Fallback local load failed for `{fallback_id}`: {e_fb!r}")
                    raise RuntimeError(
                        "Could not load the requested model or the fallback model locally.\n"
                        "This demo expects at least `runwayml/stable-diffusion-v1-5` "
                        "to be available in the local Hugging Face cache.\n"
                        "Please download/cache a compatible model once before running."
                    ) from e_fb
            else:
                # 3) Downloads allowed -> attempt to download requested model
                logger.info(
                    f"Downloads allowed. Attempting to download `{model_id}` "
                    "from the Hugging Face Hub..."
                )
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                        local_files_only=False,
                    )
                    logger.info(f"Downloaded and loaded `{model_id}` successfully.")
                except Exception as e_dl:
                    logger.error(f"Failed to download/load `{model_id}`: {e_dl!r}")
                    logger.info(f"Attempting to download fallback `{fallback_id}` as a last resort...")
                    try:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            fallback_id,
                            torch_dtype=self.torch_dtype,
                            local_files_only=False,
                        )
                        logger.info(f"Downloaded and loaded fallback `{fallback_id}` successfully.")
                        model_id = fallback_id
                    except Exception as e_final:
                        logger.error(f"Fallback download also failed for `{fallback_id}`: {e_final!r}")
                        raise RuntimeError(
                            "Failed to load the requested model and the fallback model.\n"
                            "If you are using gated models, ensure you have a valid Hugging Face token "
                            "and have accepted the model license.\n"
                            "Also check your internet connection and that the model name is correct."
                        ) from e_final

        # Optional: swap scheduler to DPMSolverMultistep for speed/quality
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            # If scheduler config is not compatible, just ignore
            pass

        # Move pipeline to device and enable basic memory optimizations
        pipe = pipe.to(self.device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            # Not all pipelines support this; ignore if not available
            pass

        # Cache and record
        self._pipelines[model_id] = pipe
        self.last_loaded_model = model_id
        return pipe

    # -------------------------------------------------------------------------
    # Public: generate images
    # -------------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        steps: int = 20,
        width: int = 512,
        height: int = 512,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        allow_download: bool = False,
    ) -> List[Image.Image]:
        """
        Generate images from a text prompt.

        Parameters:
          - prompt: text prompt (must be non-empty).
          - negative_prompt: optional negative prompt string.
          - num_images: number of images to generate (clamped to [1, 8]).
          - guidance_scale: classifier-free guidance (CFG) scale.
          - steps: number of denoising steps.
          - width, height: output resolution in pixels.
          - model_id: Hugging Face model ID to use.
          - allow_download: whether to allow network downloads for missing models.

        Returns:
          List[ PIL.Image.Image ]

        Raises:
          ValueError: if prompt is empty or invalid.
          RuntimeError: if model loading or image generation fails.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        # Simple sanity clamp to keep calls reasonable
        num_images = max(1, min(int(num_images), 8))
        steps = max(1, int(steps))

        # Load pipeline (local-first logic and fallback inside _load_pipeline)
        pipe = self._load_pipeline(model_id, allow_download=allow_download)

        images: List[Image.Image] = []

        # Prepare RNG for reproducibility (random seed, but controlled by torch.Generator)
        random_seed = torch.randint(0, 2**30, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(random_seed)
        logger.info(
            f"Generating {num_images} image(s) with seed={random_seed}, "
            f"steps={steps}, guidance_scale={guidance_scale}, size={width}x{height}"
        )

        # Run the pipeline
        try:
            outputs = pipe(
                prompt=[prompt] * num_images,
                negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                height=height,
                width=width,
            )

            # Diffusers returns an object with .images
            raw_images = getattr(outputs, "images", outputs)

            for im in raw_images:
                if isinstance(im, Image.Image):
                    images.append(im.convert("RGB"))
                else:
                    # e.g., numpy array -> attempt conversion
                    try:
                        img = Image.fromarray(im)
                        images.append(img.convert("RGB"))
                    except Exception:
                        # Fallback to pipeline helper if available
                        try:
                            pil_list = pipe.numpy_to_pil(im)
                            if pil_list:
                                images.append(pil_list[0].convert("RGB"))
                        except Exception:
                            logger.warning(
                                "Unable to convert one output to PIL.Image; skipping that result."
                            )

            if not images:
                raise RuntimeError("Pipeline returned no images.")

            logger.info(f"Generated {len(images)} image(s) successfully.")
            return images

        except Exception as gen_err:
            logger.exception("Generation call failed.")
            raise RuntimeError(f"Image generation failed: {gen_err}") from gen_err
