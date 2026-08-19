"""Stable project-facing helpers for YOLO classification predictions."""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError


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


def validate_image_bytes(image_bytes: bytes) -> None:
    """Raise a clear error when uploaded bytes are not a readable image."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("无法读取图片，请上传有效的 JPG 或 PNG 文件。") from error


def image_array_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert an uploaded image to the RGB pixel array accepted by YOLO."""

    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB")).copy()
