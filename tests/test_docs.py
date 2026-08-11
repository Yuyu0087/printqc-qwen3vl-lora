from pathlib import Path
import tomllib


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
        "printqc-paired-classification-zh-v1",
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


def test_apache_2_license_is_declared_and_present():
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert pyproject["project"]["license"] == "Apache-2.0"
