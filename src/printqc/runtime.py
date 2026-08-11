from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peft import PeftModel
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class RuntimePreflightError(RuntimeError):
    """Raised when the current machine cannot run the declared GPU mode."""


@dataclass(frozen=True)
class PrintQCRuntime:
    processor: Any
    model: Any


def preflight_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimePreflightError("CUDA is not available; v0.1 requires WSL2/Linux with NVIDIA CUDA")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }


def load_runtime(base_model: str, adapter_dir: str | Path, revision: str | None = None) -> PrintQCRuntime:
    preflight_cuda()
    processor = AutoProcessor.from_pretrained(
        adapter_dir,
        trust_remote_code=True,
        local_files_only=Path(adapter_dir).exists(),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return PrintQCRuntime(processor=processor, model=model)


def generate_pair(
    runtime: PrintQCRuntime,
    phone_image: str,
    top_image: str,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": phone_image},
                {"type": "image", "image": top_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = runtime.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = runtime.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(runtime.model.device)
    with torch.inference_mode():
        generated = runtime.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    return runtime.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
