from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


class ReleaseConfigError(ValueError):
    """Raised when the release manifest is invalid."""


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_KEYS = {
    "model_version",
    "schema_version",
    "prompt_id",
    "base_model",
    "base_revision",
    "adapter_asset_url",
    "adapter_asset_sha256",
    "adapter_asset_size",
    "adapter_tensor_sha256",
    "processor_source",
}
_ASSET_URL = (
    "https://github.com/Yuyu0087/printqc-qwen3vl-lora/releases/download/"
    "v0.1.0/printqc-qwen3vl-lora-adapter-v0.1.0.zip"
)
_TENSOR_SHA = "8076fab36711b4a5e6fcf56a73eaf988b2110346b73c6b9e46612fade621bcd0"


@dataclass(frozen=True)
class ReleaseManifest:
    model_version: str
    schema_version: str
    prompt_id: str
    base_model: str
    base_revision: str
    adapter_asset_url: str
    adapter_asset_sha256: str
    adapter_asset_size: int
    adapter_tensor_sha256: str
    processor_source: Literal["adapter", "base"]


def load_release_manifest(path: str | Path) -> ReleaseManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_release_manifest(data)


def parse_release_manifest(data: dict[str, object]) -> ReleaseManifest:
    unknown = sorted(set(data) - _EXPECTED_KEYS)
    missing = sorted(_EXPECTED_KEYS - set(data))
    if unknown:
        raise ReleaseConfigError(f"release manifest has unknown field: {unknown[0]}")
    if missing:
        raise ReleaseConfigError(f"release manifest is missing field: {missing[0]}")

    manifest = ReleaseManifest(**data)  # type: ignore[arg-type]
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: ReleaseManifest) -> None:
    if manifest.model_version != "0.1.0":
        raise ReleaseConfigError("model_version must be 0.1.0")
    if manifest.schema_version != "1.0":
        raise ReleaseConfigError("schema_version must be 1.0")
    if manifest.prompt_id != "printqc-paired-classification-zh-v1":
        raise ReleaseConfigError("prompt_id is not the reviewed prompt contract")
    if manifest.base_model != "Qwen/Qwen3-VL-4B-Instruct":
        raise ReleaseConfigError("base_model must be Qwen/Qwen3-VL-4B-Instruct")
    if not _HEX40.fullmatch(manifest.base_revision):
        raise ReleaseConfigError("base_revision must be a 40 character lowercase hex commit")
    if not _HEX64.fullmatch(manifest.adapter_asset_sha256):
        raise ReleaseConfigError("adapter_asset_sha256 must be lowercase SHA256")
    if not _HEX64.fullmatch(manifest.adapter_tensor_sha256):
        raise ReleaseConfigError("adapter_tensor_sha256 must be lowercase SHA256")
    if manifest.adapter_tensor_sha256 != _TENSOR_SHA:
        raise ReleaseConfigError("adapter_tensor_sha256 is not the reviewed v0.1.0 tensor")
    if manifest.processor_source not in {"adapter", "base"}:
        raise ReleaseConfigError("processor_source must be adapter or base")
    if not isinstance(manifest.adapter_asset_size, int) or not (0 < manifest.adapter_asset_size < 268_435_456):
        raise ReleaseConfigError("adapter_asset_size must be positive and below 256 MiB")
    _validate_asset_url(manifest.adapter_asset_url)


def _validate_asset_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ReleaseConfigError("adapter asset URL must use GitHub HTTPS")
    if "latest" in parsed.path:
        raise ReleaseConfigError("adapter asset URL must be versioned, not latest")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ReleaseConfigError("adapter asset URL must not contain credentials, query, or fragment")
    if url != _ASSET_URL:
        raise ReleaseConfigError("adapter asset URL is not the reviewed v0.1.0 GitHub Release asset")
