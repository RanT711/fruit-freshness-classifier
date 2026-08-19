"""Stable project-facing helpers for YOLO classification predictions."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


@dataclass(frozen=True)
class Prediction:
    """The normalized result shown by command-line and web interfaces."""

    label: str
    confidence: float
    message: str


def prediction_from_values(
    names: Mapping[int, str], top1: int, top1conf: float
) -> Prediction:
    """Build a user-facing prediction from YOLO's highest-scoring class."""

    label = str(names[int(top1)])
    messages = {
        "fresh": "判别结果：新鲜",
        "spoiled": "判别结果：变质",
    }
    return Prediction(
        label=label,
        confidence=float(top1conf),
        message=messages.get(label, f"判别结果：未知类别（{label}）"),
    )


def prediction_from_result(result: Any) -> Prediction:
    """Adapt one Ultralytics classification result without importing YOLO."""

    return prediction_from_values(
        result.names,
        result.probs.top1,
        result.probs.top1conf,
    )


def validate_image_dimensions(
    width: int, height: int, max_pixels: int = MAX_IMAGE_PIXELS
) -> None:
    """Reject images whose pixel count exceeds the configured safety bound."""

    if width * height > max_pixels:
        raise ValueError("图片像素过大，请上传不超过 2000 万像素的图片。")


def resolve_trusted_model_path(model_value: str, models_root: Path) -> Path:
    """Resolve a model path and keep it constrained to the trusted models folder."""

    trusted_root = models_root.resolve()
    candidate = Path(model_value).expanduser()
    if not candidate.is_absolute():
        candidate = trusted_root.parent / candidate
    candidate = candidate.resolve()

    if candidate.suffix.lower() != ".pt":
        raise ValueError("模型文件必须是 models 目录中的 .pt 文件。")
    if not candidate.is_relative_to(trusted_root):
        raise ValueError("模型文件必须位于 models 目录内。")
    return candidate


def validate_image_bytes(image_bytes: bytes) -> None:
    """Raise a clear error when uploaded bytes are not a readable image."""

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("图片文件过大，请上传不超过 10MB 的图片。")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                validate_image_dimensions(image.width, image.height)
                image.verify()
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("无法读取图片，请上传有效的 JPG 或 PNG 文件。") from error


def image_array_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert an uploaded image to the RGB pixel array accepted by YOLO."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                validate_image_dimensions(image.width, image.height)
                return np.asarray(image.convert("RGB")).copy()
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("无法读取图片，请上传有效的 JPG 或 PNG 文件。") from error
