from pathlib import Path

import pytest
from PIL import Image

from printqc.images import ImageInputError, validate_image_pair


def _image(path: Path):
    Image.new("RGB", (8, 8), "white").save(path)


def test_accepts_distinct_jpeg_png_pair(tmp_path):
    phone = tmp_path / "phone.jpg"
    top = tmp_path / "top.png"
    _image(phone)
    _image(top)

    result = validate_image_pair(phone, top)

    assert result.phone_image == phone
    assert result.top_image == top


def test_rejects_duplicate_paths(tmp_path):
    phone = tmp_path / "same.jpg"
    _image(phone)

    with pytest.raises(ImageInputError, match="different"):
        validate_image_pair(phone, phone)


def test_rejects_webp_for_v0_1(tmp_path):
    phone = tmp_path / "phone.webp"
    top = tmp_path / "top.png"
    phone.write_bytes(b"not used")
    _image(top)

    with pytest.raises(ImageInputError, match="JPEG or PNG"):
        validate_image_pair(phone, top)
