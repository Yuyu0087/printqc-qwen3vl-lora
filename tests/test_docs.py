from pathlib import Path


def test_readme_states_public_scope_without_overclaiming():
    readme = Path("README.md").read_text(encoding="utf-8")

    required = [
        "WSL2/Linux",
        "Python 3.11",
        "RTX 5070",
        "torch 2.12.0+cu130",
        "CUDA 13.0",
        "bitsandbytes 0.50.0",
        "GitHub Release",
        "not hosted on Hugging Face",
        "phone side-view image first",
        "top-down printer image second",
        "fail",
        "printer control",
    ]
    for phrase in required:
        assert phrase in readme


def test_readme_does_not_claim_accuracy_or_native_windows():
    readme = Path("README.md").read_text(encoding="utf-8").lower()

    banned = ["accuracy: 0.", "f1:", "supports native windows", "supports cpu"]
    for phrase in banned:
        assert phrase not in readme
