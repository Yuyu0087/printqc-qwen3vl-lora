from __future__ import annotations

import argparse
import json
from pathlib import Path

from printqc.inference import run_inference
from printqc.parsing import abstained_result, parse_model_output, ParseError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local smoke inference on a gold_pilot sample.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[args.sample_index]
    raw = run_inference(
        phone_image=row["phone_path_wsl"],
        top_image=row["top_path_wsl"],
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_new_tokens=128,
    )
    try:
        parsed = parse_model_output(raw)
    except ParseError as exc:
        parsed = abstained_result(str(exc))
    report = {
        "sample_uid": row.get("sample_uid"),
        "human_label": row.get("human_label"),
        "human_severity": row.get("severity"),
        "phone_sha256": row.get("phone_sha256"),
        "top_sha256": row.get("top_sha256"),
        "raw_text": raw,
        "parsed": parsed,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parsed, ensure_ascii=False, sort_keys=True))
    return 0 if not parsed.get("abstained") else 5


if __name__ == "__main__":
    raise SystemExit(main())
