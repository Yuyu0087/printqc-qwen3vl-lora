from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable
from urllib.parse import urlparse
import zipfile

import httpx
from platformdirs import user_cache_dir

from printqc.config import ReleaseManifest


class ArtifactDownloadError(RuntimeError):
    """Raised when the adapter cannot be downloaded."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when downloaded or cached adapter bytes fail validation."""


def ensure_adapter(
    manifest: ReleaseManifest,
    cache_root: str | Path | None = None,
    offline: bool = False,
    fetcher: Callable[[str], bytes] | None = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else Path(user_cache_dir("printqc", "Yuyu0087"))
    adapter_dir = root / manifest.model_version / manifest.adapter_asset_sha256
    ready = adapter_dir / "READY"
    adapter_file = adapter_dir / "adapter_model.safetensors"

    if ready.exists() and adapter_file.exists():
        _validate_cached_adapter(adapter_dir, manifest)
        return adapter_dir
    if offline:
        raise ArtifactDownloadError("adapter cache is missing and offline mode is enabled")

    root.mkdir(parents=True, exist_ok=True)
    body = fetcher(manifest.adapter_asset_url) if fetcher else _download_bytes(manifest)
    _validate_asset_body(body, manifest)

    with tempfile.TemporaryDirectory(prefix="printqc-adapter-", dir=root) as tmp:
        tmp_dir = Path(tmp)
        asset_path = tmp_dir / "asset.zip"
        asset_path.write_bytes(body)
        _extract_zip(asset_path, tmp_dir / "extract")
        _validate_cached_adapter(tmp_dir / "extract", manifest)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_dir / "extract", adapter_dir)
    ready.write_text(manifest.adapter_asset_sha256, encoding="utf-8")
    return adapter_dir


def _download_bytes(manifest: ReleaseManifest) -> bytes:
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            response = client.get(manifest.adapter_asset_url)
            for item in response.history + [response]:
                _validate_redirect_host(str(item.url))
            response.raise_for_status()
            body = response.content
    except httpx.HTTPError as exc:
        raise ArtifactDownloadError(f"adapter download failed: {exc.__class__.__name__}") from exc
    return body


def _validate_redirect_host(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (host == "github.com" or host.endswith(".githubusercontent.com")):
        raise ArtifactDownloadError("adapter download redirected outside approved GitHub HTTPS hosts")


def _validate_asset_body(body: bytes, manifest: ReleaseManifest) -> None:
    if len(body) != manifest.adapter_asset_size:
        raise ArtifactIntegrityError("adapter asset size does not match manifest")
    digest = hashlib.sha256(body).hexdigest()
    if digest != manifest.adapter_asset_sha256:
        raise ArtifactIntegrityError("adapter asset SHA256 does not match manifest")


def _extract_zip(asset_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset_path) as zf:
        seen_lower: set[str] = set()
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if info.is_dir():
                continue
            if name.startswith("/") or ".." in parts:
                raise ArtifactIntegrityError("unsafe ZIP member path")
            lowered = name.lower()
            if lowered in seen_lower:
                raise ArtifactIntegrityError("unsafe ZIP case collision")
            seen_lower.add(lowered)
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _validate_cached_adapter(adapter_dir: Path, manifest: ReleaseManifest) -> None:
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise ArtifactIntegrityError("adapter_model.safetensors is missing from adapter cache")
    digest = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    if digest != manifest.adapter_tensor_sha256:
        raise ArtifactIntegrityError("adapter tensor SHA256 does not match manifest")
