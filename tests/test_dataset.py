from pathlib import Path

import pytest
from PIL import Image

from fruit_grader.dataset import DatasetSplit, prepare_classification_dataset


def make_image(path: Path) -> None:
    Image.new("RGB", (8, 8), "red").save(path)


def test_prepare_dataset_creates_reproducible_stratified_split(tmp_path: Path):
    raw = tmp_path / "raw"
    for label in ("fresh", "spoiled"):
        folder = raw / label
        folder.mkdir(parents=True)
        for index in range(10):
            make_image(folder / f"{index}.jpg")

    summary = prepare_classification_dataset(
        raw, tmp_path / "processed-a", DatasetSplit(60, 20, 20), seed=7
    )
    prepare_classification_dataset(
        raw, tmp_path / "processed-b", DatasetSplit(60, 20, 20), seed=7
    )

    assert summary == {
        "fresh": {"train": 6, "val": 2, "test": 2},
        "spoiled": {"train": 6, "val": 2, "test": 2},
    }
    for label in ("fresh", "spoiled"):
        assert sorted(
            path.name for path in (tmp_path / "processed-a" / "test" / label).iterdir()
        ) == sorted(
            path.name for path in (tmp_path / "processed-b" / "test" / label).iterdir()
        )


def test_prepare_dataset_rejects_missing_class(tmp_path: Path):
    (tmp_path / "raw" / "fresh").mkdir(parents=True)

    with pytest.raises(ValueError, match="spoiled"):
        prepare_classification_dataset(tmp_path / "raw", tmp_path / "processed")


def test_dataset_split_rejects_percentages_not_totaling_one_hundred():
    with pytest.raises(ValueError, match="100"):
        DatasetSplit(70, 20, 20).validate()
