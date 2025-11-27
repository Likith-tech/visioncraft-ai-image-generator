# utils.py
"""
Utility functions for:
- Basic safety filtering for text prompts.
- Style-based prompt engineering for image generation.
"""

from typing import Dict, Set

# -------------------------------------------------------------------------
# Basic unsafe / banned word list
# -------------------------------------------------------------------------
# NOTE:
# This is a simple keyword-based filter to demonstrate responsible use.
# It is NOT a production-grade safety system, but is sufficient for this internship task.
UNSAFE_WORDS: Set[str] = {
    "nude", "nudity", "nsfw",
    "violence", "blood", "gore",
    "kill", "murder", "weapon", "gun",
    "sexual", "sex", "porn", "explicit",
    "rape", "abuse",
    "drugs", "cocaine", "heroin",
    "suicide", "self-harm", "self harm",
    "racist", "hate", "slur",
}


def is_safe_prompt(prompt: str) -> bool:
    """
    Basic safety checker for text prompts.

    Returns:
        True  -> if the prompt does NOT contain any banned keywords.
        False -> if the prompt contains at least one banned keyword,
                 or if the prompt is not a string.

    This is a simple case-insensitive substring check.
    """
    if not isinstance(prompt, str):
        return False

    text = prompt.lower()

    for bad_word in UNSAFE_WORDS:
        if bad_word in text:
            return False

    return True


# -------------------------------------------------------------------------
# Style prompt engineering
# -------------------------------------------------------------------------
STYLE_TOKENS: Dict[str, str] = {
    "Default": "",
    "Photorealistic": (
        "ultra realistic, 8k, high detail, DSLR, cinematic lighting, "
        "sharp focus, highly detailed, professional photography"
    ),
    "Artistic": (
        "beautiful art, painterly, expressive brush strokes, artistic, "
        "vivid colors, concept art, illustration"
    ),
    "Cartoon": (
        "cartoon style, cel-shaded, bright simple colors, thick outlines, "
        "2d illustration, clean lines"
    ),
    "Cyberpunk": (
        "cyberpunk, neon glow, futuristic city, neon signs, sci-fi lighting, "
        "rainy streets, reflections"
    ),
    "VanGogh": (
        "Van Gogh style, impressionist brush strokes, swirling patterns, "
        "strong texture, vibrant contrasting colors"
    ),
}


def apply_style_to_prompt(prompt: str, style: str) -> str:
    """
    Appends style-specific tokens to a prompt.

    Args:
        prompt: Base user prompt (string).
        style:  One of the keys in STYLE_TOKENS. If style is unknown
                or 'Default', returns the original prompt.

    Returns:
        The resulting prompt with style tokens appended, e.g.:
          "a cat in a field, ultra realistic, 8k, high detail, ..."
    """
    if not isinstance(prompt, str) or not prompt.strip():
        # If prompt is invalid, just return it unchanged
        return prompt

    style_suffix = STYLE_TOKENS.get(style, "").strip()

    if style_suffix:
        return f"{prompt}, {style_suffix}"

    return prompt
