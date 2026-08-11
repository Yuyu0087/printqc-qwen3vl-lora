from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from printqc.artifacts import ensure_adapter
from printqc.config import load_release_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PrintQC release manifest and adapter asset.")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()

    asset = Path(args.asset)
    manifest = load_release_manifest(args.manifest)
    if asset.stat().st_size != manifest.adapter_asset_size:
        raise SystemExit("asset size does not match release manifest")
    if sha256_file(asset) != manifest.adapter_asset_sha256:
        raise SystemExit("asset SHA256 does not match release manifest")
    cache_root = Path(args.cache_root) if args.cache_root else asset.parent / "_validate_cache"
    ensure_adapter(manifest, cache_root=cache_root, fetcher=lambda _url: asset.read_bytes())
    print("release asset validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
