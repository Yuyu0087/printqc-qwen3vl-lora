from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImageInputError(ValueError):
    """Raised when input images are missing, unsafe, or unsupported."""


@dataclass(frozen=True)
class ImagePair:
    phone_image: Path
    top_image: Path


def validate_image_pair(phone_image: str | Path, top_image: str | Path) -> ImagePair:
    phone = Path(phone_image)
    top = Path(top_image)
    if phone == top or phone.resolve() == top.resolve():
        raise ImageInputError("phone and top images must be different files")
    _validate_one(phone)
    _validate_one(top)
    return ImagePair(phone, top)


def _validate_one(path: Path) -> None:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ImageInputError("v0.1 accepts JPEG or PNG images only")
    if not path.is_file():
        raise ImageInputError(f"image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageInputError(f"image file is not readable: {path}") from exc
