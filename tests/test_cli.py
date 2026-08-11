import json

from PIL import Image

from printqc.cli import main


def test_cli_mock_inference_writes_result(tmp_path, monkeypatch):
    phone = tmp_path / "phone.jpg"
    top = tmp_path / "top.jpg"
    output = tmp_path / "result.json"
    Image.new("RGB", (8, 8), "white").save(phone)
    Image.new("RGB", (8, 8), "black").save(top)

    monkeypatch.setattr(
        "printqc.cli.run_inference",
        lambda **_kwargs: '{"label":"normal","severity":0,"confidence":0.8,"evidence":"smooth surface"}',
    )

    code = main(["--phone-image", str(phone), "--top-image", str(top), "--output", str(output)])

    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["label"] == "normal"
    assert data["abstained"] is False


def test_cli_parse_failure_exits_5(tmp_path, monkeypatch):
    phone = tmp_path / "phone.jpg"
    top = tmp_path / "top.jpg"
    output = tmp_path / "result.json"
    Image.new("RGB", (8, 8), "white").save(phone)
    Image.new("RGB", (8, 8), "black").save(top)
    monkeypatch.setattr("printqc.cli.run_inference", lambda **_kwargs: "bad")

    code = main(["--phone-image", str(phone), "--top-image", str(top), "--output", str(output)])

    assert code == 5
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["label"] is None
    assert data["abstained"] is True
