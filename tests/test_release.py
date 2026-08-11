import hashlib
import json
import zipfile

from scripts.build_release_asset import build_release_asset, stage_adapter
from scripts.validate_release import scan_public_repo


def _source_adapter(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    private_base = "/" + "home" + "/user/private/base"
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": private_base, "r": 8}),
        encoding="utf-8",
    )
    (source / "processor_config.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "chat_template.jinja").write_text("template", encoding="utf-8")
    (source / "trainer_state.json").write_text("private", encoding="utf-8")
    return source


def test_stage_adapter_sanitizes_private_base_path(tmp_path):
    source = _source_adapter(tmp_path)
    stage = tmp_path / "stage"

    manifest = stage_adapter(source, stage, adapter_sha256=hashlib.sha256(b"adapter").hexdigest())

    config = json.loads((stage / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert not (stage / "trainer_state.json").exists()
    assert manifest["files"]["adapter_model.safetensors"]["sha256"] == hashlib.sha256(b"adapter").hexdigest()


def test_build_release_asset_is_deterministic(tmp_path):
    source = _source_adapter(tmp_path)
    stage = tmp_path / "stage"
    asset_a = tmp_path / "a.zip"
    asset_b = tmp_path / "b.zip"
    adapter_sha = hashlib.sha256(b"adapter").hexdigest()

    stage_adapter(source, stage, adapter_sha256=adapter_sha)
    build_release_asset(stage, asset_a)
    build_release_asset(stage, asset_b)

    assert asset_a.read_bytes() == asset_b.read_bytes()
    with zipfile.ZipFile(asset_a) as zf:
        assert sorted(zf.namelist()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
            "artifact_manifest.json",
            "chat_template.jinja",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]


def test_public_repo_scan_redacts_private_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    private_path = "/" + "home" + "/zzzero1/models/Qwen3-VL-4B-Instruct"
    (repo / "README.md").write_text(
        f"cache path: {private_path}\n",
        encoding="utf-8",
    )

    findings = scan_public_repo(repo)

    assert findings
    assert findings[0]["rule"] == "private-posix-path"
    assert findings[0]["file"] == "README.md"
    assert private_path not in json.dumps(findings)
