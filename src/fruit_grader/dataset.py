"""Utilities for preparing a reproducible YOLO classification dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

from PIL import Image, UnidentifiedImageError


LABELS = ("fresh", "spoiled")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetSplit:
    """Percentage allocation for train, validation, and test images."""

    train: int = 70
    val: int = 15
    test: int = 15

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value < 0 for value in values):
            raise ValueError("训练、验证和测试比例不能为负数。")
        if sum(values) != 100:
            raise ValueError("训练、验证和测试比例之和必须为 100。")


def prepare_classification_dataset(
    raw_root: Path,
    output_root: Path,
    split: DatasetSplit = DatasetSplit(),
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Copy valid source images into a reproducible class-based split layout."""

    split.validate()
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    _ensure_safe_output_root(output_root)

    images_by_label = {
        label: _valid_images(raw_root / label) for label in LABELS
    }
    _validate_sources(raw_root, images_by_label)

    if output_root.exists():
        shutil.rmtree(output_root)

    summary: dict[str, dict[str, int]] = {}
    for label, images in images_by_label.items():
        shuffled = list(images)
        random.Random(f"{seed}:{label}").shuffle(shuffled)
        boundaries = _split_boundaries(len(shuffled), split)
        groups = {
            "train": shuffled[: boundaries[0]],
            "val": shuffled[boundaries[0] : boundaries[1]],
            "test": shuffled[boundaries[1] :],
        }

        summary[label] = {}
        for subset, files in groups.items():
            target = output_root / subset / label
            target.mkdir(parents=True, exist_ok=True)
            for source in files:
                shutil.copy2(source, target / source.name)
            summary[label][subset] = len(files)

    return summary


def _ensure_safe_output_root(output_root: Path) -> None:
    if output_root == Path(output_root.anchor):
        raise ValueError("输出目录不能是文件系统根目录。")


def _validate_sources(
    raw_root: Path, images_by_label: dict[str, list[Path]]
) -> None:
    if not raw_root.is_dir():
        raise ValueError(f"原始数据目录不存在：{raw_root}")
    for label in LABELS:
        class_dir = raw_root / label
        if not class_dir.is_dir():
            raise ValueError(f"缺少类别目录：{label}")
    for label in LABELS:
        if not images_by_label[label]:
            raise ValueError(f"类别目录没有可用图片：{label}")


def _valid_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []

    valid: list[Path] = []
    for path in sorted(class_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            continue
        valid.append(path)
    return valid


def _split_boundaries(count: int, split: DatasetSplit) -> tuple[int, int]:
    train_end = count * split.train // 100
    val_end = train_end + count * split.val // 100
    return train_end, val_end
