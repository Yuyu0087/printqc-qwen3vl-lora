from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


PUBLIC_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
FIXED_ZIP_TIME = (2026, 8, 11, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_adapter(source: str | Path, stage: str | Path, adapter_sha256: str) -> dict:
    source = Path(source)
    stage = Path(stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    manifest = {"base_model": BASE_MODEL, "files": {}}

    for name in PUBLIC_FILES:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(f"required adapter file missing: {src}")
        dst = stage / name
        if name == "adapter_config.json":
            data = json.loads(src.read_text(encoding="utf-8"))
            data["base_model_name_or_path"] = BASE_MODEL
            dst.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            shutil.copy2(src, dst)

    tensor_sha = sha256_file(stage / "adapter_model.safetensors")
    if tensor_sha != adapter_sha256:
        raise ValueError("adapter_model.safetensors SHA256 does not match the reviewed tensor")

    for path in sorted(stage.iterdir()):
        if path.is_file():
            manifest["files"][path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    (stage / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_release_asset(stage: str | Path, asset_path: str | Path) -> dict:
    stage = Path(stage)
    asset_path = Path(asset_path)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in stage.iterdir() if path.is_file())
    with zipfile.ZipFile(asset_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in names:
            source = stage / name
            info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, source.read_bytes(), compresslevel=9)
    return {"size": asset_path.stat().st_size, "sha256": sha256_file(asset_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic PrintQC adapter release asset.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    args = parser.parse_args()

    stage_adapter(args.source, args.stage, args.adapter_sha256)
    asset = build_release_asset(args.stage, args.asset)
    print(json.dumps(asset, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
