from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from fruit_grader import inference
from fruit_grader.inference import (
    prediction_from_values,
    resolve_trusted_model_path,
    validate_image_bytes,
    validate_image_dimensions,
)


def test_fresh_prediction_uses_chinese_fresh_message():
    prediction = prediction_from_values({0: "fresh", 1: "spoiled"}, 0, 0.875)

    assert prediction.label == "fresh"
    assert prediction.confidence == 0.875
    assert prediction.message == "判别结果：新鲜"


def test_unknown_prediction_is_not_mislabeled():
    prediction = prediction_from_values({0: "unknown"}, 0, 0.5)

    assert prediction.message == "判别结果：未知类别（unknown）"


def test_validate_image_bytes_accepts_png():
    output = BytesIO()
    Image.new("RGB", (8, 8), "green").save(output, format="PNG")

    assert validate_image_bytes(output.getvalue()) is None


def test_image_array_from_bytes_converts_uploaded_png_to_rgb_pixels():
    output = BytesIO()
    Image.new("RGBA", (3, 2), (12, 34, 56, 128)).save(output, format="PNG")

    image = inference.image_array_from_bytes(output.getvalue())

    assert image.shape == (2, 3, 3)
    assert image[0, 0].tolist() == [12, 34, 56]


def test_validate_image_bytes_rejects_non_image():
    with pytest.raises(ValueError, match="无法读取图片"):
        validate_image_bytes(b"not an image")


def test_validate_image_dimensions_rejects_oversized_image():
    with pytest.raises(ValueError, match="像素"):
        validate_image_dimensions(5_000, 4_001)


def test_resolve_trusted_model_path_rejects_path_outside_models(tmp_path: Path):
    models_root = tmp_path / "models"
    models_root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a model")

    with pytest.raises(ValueError, match="models"):
        resolve_trusted_model_path(str(outside), models_root)
