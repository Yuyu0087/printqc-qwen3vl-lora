from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from printqc.artifacts import ArtifactDownloadError, ArtifactIntegrityError
from printqc.images import ImageInputError
from printqc.inference import run_inference
from printqc.parsing import ParseError, abstained_result, parse_model_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PrintQC paired-image classification.")
    parser.add_argument("--phone-image", required=True, help="Path to the phone side-view image.")
    parser.add_argument("--top-image", required=True, help="Path to the top-down printer image.")
    parser.add_argument("--output", required=True, help="Path to write result JSON.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct", help="Base model ID or local path.")
    parser.add_argument("--base-revision", default=None, help="Pinned Hugging Face base revision.")
    parser.add_argument("--adapter-dir", default=None, help="Local extracted PEFT adapter directory.")
    parser.add_argument("--manifest", default=None, help="Release manifest path for automatic adapter download.")
    parser.add_argument("--cache-dir", default=None, help="PrintQC cache directory.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--offline", action="store_true", help="Use only already cached model assets.")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        raw = run_inference(
            phone_image=args.phone_image,
            top_image=args.top_image,
            offline=args.offline,
            base_model=args.base_model,
            base_revision=args.base_revision,
            adapter_dir=args.adapter_dir,
            manifest_path=args.manifest,
            cache_root=args.cache_dir,
            max_new_tokens=args.max_new_tokens,
        )
        result = parse_model_output(raw)
        _write_json(output, result)
        return 0
    except ImageInputError as exc:
        _write_json(output, abstained_result(str(exc)))
        return 2
    except (ArtifactDownloadError, ArtifactIntegrityError) as exc:
        _write_json(output, abstained_result(str(exc)))
        return 3
    except ParseError as exc:
        _write_json(output, abstained_result(str(exc)))
        return 5


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
