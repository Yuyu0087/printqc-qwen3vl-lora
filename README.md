# PrintQC Qwen3-VL LoRA

Experimental paired-image 3D-print under-extrusion classification with Qwen3-VL and a small LoRA adapter.

This project is a GitHub-only release package. The program downloads the official base model from Hugging Face (`Qwen/Qwen3-VL-4B-Instruct`) and downloads the trained adapter from this repository's fixed `v0.1.0` GitHub Release asset. The adapter is not hosted on Hugging Face.

## Scope

- Status: experimental v0.1.0.
- Supported runtime: WSL2/Linux, Python 3.11, NVIDIA CUDA.
- Tested local stack: RTX 5070, torch 2.12.0+cu130, CUDA 13.0, bitsandbytes 0.50.0.
- Task: paired-image classification for `normal`, `under_extrusion`, or `unsure`.
- Not included: printer control, Moonraker/Klipper integration, G-code commands, webcam streaming, accuracy/F1 claims, native Windows support, CPU support, or private training images.

## Install

```bash
git clone https://github.com/Yuyu0087/printqc-qwen3vl-lora.git
cd printqc-qwen3vl-lora
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cu130.txt
```

First use may require about 12 GiB of steady-state cache space for the base model and adapter. Use an ext4 WSL path for `HF_HOME` and `PRINTQC_CACHE_DIR`; using `/mnt/*` can copy cache blobs instead of symlinking and may temporarily need much more disk space.

## Run

Use a phone side-view image first and a top-down printer image second:

```bash
export PRINTQC_CACHE_DIR="$HOME/.cache/printqc"
printqc-infer \
  --phone-image ./examples/phone.jpg \
  --top-image ./examples/top.jpg \
  --manifest ./src/printqc/resources/release_manifest.json \
  --output result.json
```

Output schema:

```json
{
  "label": "under_extrusion",
  "severity": 2,
  "confidence": 0.73,
  "evidence": "visible gaps between adjacent extrusion paths",
  "abstained": false
}
```

If output parsing fails, the program fails closed with exit code 5 and writes `label: null` plus `abstained: true`. Image input errors return exit code 2. Missing offline cache returns an adapter/base cache error.

## Offline Use

After one successful online run, use:

```bash
printqc-infer --offline --phone-image ./phone.jpg --top-image ./top.jpg --manifest ./src/printqc/resources/release_manifest.json --output result.json
```

Offline mode never downloads missing assets. If the adapter or base model cache is incomplete, the command fails instead of silently changing behavior.

## Training

Training materials are in `training/`. v0.1.0 provides a minimal dataset example and a LLaMA-Factory-style LoRA configuration template. Private source photos and exact resume checkpoints are intentionally not included in the public repository.

## Prompt

The public interface is English. The internal prompt contract is fixed as `printqc-paired-classification-zh-v1` for reproducibility of this trained adapter.

## v0.1.0 Verification

Before the public v0.1.0 release, the maintainer verified:

- The GitHub Release asset round-trip passes.
- The pinned Hugging Face base revision is proven equivalent to the trained local ModelScope base.
- A real first-download Hugging Face base inference passes while the repository is still private.
- Apache-2.0 licensing is explicitly approved.
