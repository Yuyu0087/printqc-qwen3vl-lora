import hashlib
import io
import json
import zipfile

import pytest

from printqc.artifacts import ArtifactIntegrityError, ensure_adapter
from printqc.config import ReleaseConfigError, ReleaseManifest, load_release_manifest


def _valid_manifest(tmp_path, **overrides):
    data = json.loads((tmp_path / "manifest.json").read_text()) if (tmp_path / "manifest.json").exists() else {
        "model_version": "0.1.0",
        "schema_version": "1.0",
        "prompt_id": "printqc-paired-classification-zh-v1",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "base_revision": "0123456789abcdef0123456789abcdef01234567",
        "adapter_asset_url": "https://github.com/Yuyu0087/printqc-qwen3vl-lora/releases/download/v0.1.0/printqc-qwen3vl-lora-adapter-v0.1.0.zip",
        "adapter_asset_sha256": "a" * 64,
        "adapter_asset_size": 128,
        "adapter_tensor_sha256": "8076fab36711b4a5e6fcf56a73eaf988b2110346b73c6b9e46612fade621bcd0",
        "processor_source": "adapter",
    }
    data.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_release_manifest(path)


def _artifact_manifest(**overrides):
    data = {
        "model_version": "0.1.0",
        "schema_version": "1.0",
        "prompt_id": "printqc-paired-classification-zh-v1",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "base_revision": "0123456789abcdef0123456789abcdef01234567",
        "adapter_asset_url": "https://github.com/Yuyu0087/printqc-qwen3vl-lora/releases/download/v0.1.0/printqc-qwen3vl-lora-adapter-v0.1.0.zip",
        "adapter_asset_sha256": "a" * 64,
        "adapter_asset_size": 128,
        "adapter_tensor_sha256": "8076fab36711b4a5e6fcf56a73eaf988b2110346b73c6b9e46612fade621bcd0",
        "processor_source": "adapter",
    }
    data.update(overrides)
    return ReleaseManifest(**data)


def _zip_bytes(inner_name="adapter_model.safetensors", payload=b"adapter"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, payload)
    return buf.getvalue()


def test_manifest_rejects_latest_url(tmp_path):
    with pytest.raises(ReleaseConfigError, match="versioned"):
        _valid_manifest(
            tmp_path,
            adapter_asset_url="https://github.com/Yuyu0087/printqc-qwen3vl-lora/releases/latest/download/printqc-qwen3vl-lora-adapter-v0.1.0.zip",
        )


def test_manifest_rejects_unknown_field(tmp_path):
    with pytest.raises(ReleaseConfigError, match="unknown"):
        _valid_manifest(tmp_path, extra="nope")


def test_manifest_accepts_valid_fixture():
    manifest = load_release_manifest("tests/fixtures/release_manifest.valid.json")

    assert manifest.base_model == "Qwen/Qwen3-VL-4B-Instruct"
    assert manifest.processor_source == "adapter"


def test_wrong_asset_hash_never_becomes_cache(tmp_path):
    body = _zip_bytes(payload=b"wrong")
    manifest = _artifact_manifest(adapter_asset_size=len(body), adapter_asset_sha256="0" * 64)

    with pytest.raises(ArtifactIntegrityError, match="SHA256"):
        ensure_adapter(manifest, cache_root=tmp_path, fetcher=lambda _url: body)

    assert not list(tmp_path.rglob("READY"))


def test_valid_asset_extracts_and_marks_ready(tmp_path):
    payload = b"adapter"
    body = _zip_bytes(payload=payload)
    manifest = _artifact_manifest(
        adapter_asset_size=len(body),
        adapter_asset_sha256=hashlib.sha256(body).hexdigest(),
        adapter_tensor_sha256=hashlib.sha256(payload).hexdigest(),
    )

    adapter_dir = ensure_adapter(manifest, cache_root=tmp_path, fetcher=lambda _url: body)

    assert (adapter_dir / "adapter_model.safetensors").read_bytes() == payload
    assert (adapter_dir / "READY").read_text(encoding="utf-8") == manifest.adapter_asset_sha256


def test_zip_slip_is_rejected(tmp_path):
    body = _zip_bytes(inner_name="../adapter_model.safetensors")
    manifest = _artifact_manifest(
        adapter_asset_size=len(body),
        adapter_asset_sha256=hashlib.sha256(body).hexdigest(),
    )

    with pytest.raises(ArtifactIntegrityError, match="unsafe"):
        ensure_adapter(manifest, cache_root=tmp_path, fetcher=lambda _url: body)
