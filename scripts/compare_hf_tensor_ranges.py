from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import urllib.request


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def read_remote_header(url: str) -> dict:
    prefix = fetch_range(url, 0, 7)
    header_len = struct.unpack("<Q", prefix[:8])[0]
    header_bytes = fetch_range(url, 8, 8 + header_len - 1)
    return json.loads(header_bytes)


def read_local_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return header_len, json.loads(handle.read(header_len))


def local_range(path: Path, start: int, end: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(end - start + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare small HF safetensors byte ranges to local ModelScope shards.")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-base", required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    local_base = Path(args.local_base)
    index = json.loads((local_base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for tensor, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(tensor)

    checked = []
    for shard, tensors in sorted(by_shard.items()):
        url = f"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/{args.revision}/{shard}"
        local_path = local_base / shard
        local_header_len, local_header = read_local_header(local_path)
        remote_header = read_remote_header(url)
        if remote_header != local_header:
            raise SystemExit(f"header mismatch for {shard}")
        data_start = 8 + local_header_len
        for tensor in sorted(tensors)[: args.samples]:
            offsets = local_header[tensor]["data_offsets"]
            start = data_start + offsets[0]
            end = min(data_start + offsets[1] - 1, start + 4095)
            remote = fetch_range(url, start, end)
            local = local_range(local_path, start, end)
            if remote != local:
                raise SystemExit(f"tensor range mismatch: {shard}:{tensor}")
            checked.append({"shard": shard, "tensor": tensor, "start": start, "end": end, "bytes": len(local)})
    print(json.dumps({"checked_ranges": checked}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
