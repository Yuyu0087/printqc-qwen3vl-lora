from __future__ import annotations

from pathlib import Path

from printqc.artifacts import ensure_adapter
from printqc.config import load_release_manifest
from printqc.images import validate_image_pair
from printqc.prompts import PAIRED_CLASSIFICATION_PROMPT
from printqc.runtime import generate_pair, load_runtime


def run_inference(
    phone_image: str | Path,
    top_image: str | Path,
    offline: bool = False,
    prompt: str = PAIRED_CLASSIFICATION_PROMPT,
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    base_revision: str | None = None,
    adapter_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    max_new_tokens: int = 256,
) -> str:
    pair = validate_image_pair(phone_image, top_image)
    if adapter_dir is None:
        if manifest_path is None:
            raise RuntimeError("adapter_dir or manifest_path is required until the v0.1 manifest is generated")
        manifest = load_release_manifest(manifest_path)
        adapter_dir = ensure_adapter(manifest, cache_root=cache_root, offline=offline)
        base_model = manifest.base_model
        base_revision = manifest.base_revision
    runtime = load_runtime(base_model=base_model, adapter_dir=adapter_dir, revision=base_revision)
    return generate_pair(
        runtime,
        phone_image=str(pair.phone_image.resolve()),
        top_image=str(pair.top_image.resolve()),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
