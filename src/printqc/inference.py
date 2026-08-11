from __future__ import annotations

from pathlib import Path

from printqc.images import validate_image_pair
from printqc.prompts import PAIRED_CLASSIFICATION_PROMPT


def run_inference(
    phone_image: str | Path,
    top_image: str | Path,
    offline: bool = False,
    prompt: str = PAIRED_CLASSIFICATION_PROMPT,
) -> str:
    validate_image_pair(phone_image, top_image)
    raise RuntimeError(
        "real model inference is not wired yet; this private scaffold only supports mocked CLI tests"
    )
