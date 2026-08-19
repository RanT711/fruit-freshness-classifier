# 水果新鲜度离线判别系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个使用 YOLO 分类模型判断本地水果图片“新鲜/变质”的可训练、可预测、可上传展示项目。

**Architecture:** 项目将原始图片整理成 Ultralytics 分类目录；`dataset.py` 只负责验证与可复现切分，`inference.py` 只负责标准化预测结果。训练、单图预测和网页上传分别通过独立入口调用这些模块，防止页面逻辑与模型逻辑耦合。

**Tech Stack:** Python 3.10+、Ultralytics YOLO、Pillow、Streamlit、pytest。

**Spec:** `docs/superpowers/specs/2026-08-19-fruit-freshness-classifier-design.md`

## Global Constraints

- 只做本地静态图片分类，不接入摄像头、视频流或多目标检测。
- 标签固定为 `fresh` 与 `spoiled`，页面显示对应中文结论。
- 默认模型为 `yolo26n-cls.pt`，但所有入口允许以参数覆盖模型路径。
- `data/`、`models/` 与 `runs/` 不提交到 Git；只提交目录说明和代码。
- 每个新增生产函数必须先有一个实际失败的 pytest 测试。
- 训练和页面运行不应依赖测试数据或真实模型权重。

---

## File Structure

```text
.gitignore
README.md
requirements.txt
app.py
train.py
predict.py
data/README.md
scripts/prepare_dataset.py
src/fruit_grader/__init__.py
src/fruit_grader/dataset.py
src/fruit_grader/inference.py
tests/conftest.py
tests/test_dataset.py
tests/test_inference.py
tests/test_cli.py
```

## Task 1: Project scaffold and dependency contract

**Files:**

- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/fruit_grader/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_cli.py`
- Create: `data/README.md`

**Interfaces:**

- Produces importable package `fruit_grader`.
- Produces `data/README.md` with exact `data/raw/{fresh,spoiled}` source layout.

- [ ] **Step 1: Write the failing package-import test**

```python
# tests/test_cli.py
def test_package_can_be_imported():
    import fruit_grader

    assert fruit_grader.__name__ == "fruit_grader"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli.py::test_package_can_be_imported -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fruit_grader'`.

- [ ] **Step 3: Create the minimum package and test configuration**

```ini
# pytest.ini
[pytest]
pythonpath = src
testpaths = tests
```

```python
# src/fruit_grader/__init__.py
"""Offline fruit-freshness classification package."""
```

Add `pytest.ini` to the file structure if it is not already present. Add a `requirements.txt` containing exactly `ultralytics`, `streamlit`, `Pillow`, and `pytest`. Add `.gitignore` entries for `.venv/`, `__pycache__/`, `.pytest_cache/`, `data/raw/`, `data/processed/`, `models/*.pt`, and `runs/`. Add `data/README.md` documenting the two raw class folders.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_cli.py::test_package_can_be_imported -v`

Expected: PASS.

- [ ] **Step 5: Commit the scaffold**

```bash
git add .gitignore pytest.ini requirements.txt data/README.md src/fruit_grader/__init__.py tests/conftest.py tests/test_cli.py
git commit -m "chore: scaffold fruit classifier project"
```

## Task 2: Reproducible classification dataset preparation

**Files:**

- Create: `src/fruit_grader/dataset.py`
- Create: `tests/test_dataset.py`
- Create: `scripts/prepare_dataset.py`

**Interfaces:**

- Consumes: `data/raw/fresh/` and `data/raw/spoiled/` directories containing image files.
- Produces: `prepare_classification_dataset(raw_root: Path, output_root: Path, split: DatasetSplit = DatasetSplit(70, 15, 15), seed: int = 42) -> dict[str, dict[str, int]]`.
- Produces: processed directories `train|val|test/fresh|spoiled` and a per-split file count summary.

- [ ] **Step 1: Write failing tests for valid images, split counts, and missing classes**

```python
# tests/test_dataset.py
from pathlib import Path

import pytest
from PIL import Image

from fruit_grader.dataset import DatasetSplit, prepare_classification_dataset


def make_image(path: Path) -> None:
    Image.new("RGB", (8, 8), "red").save(path)


def test_prepare_dataset_creates_stratified_split(tmp_path: Path):
    raw = tmp_path / "raw"
    for label in ("fresh", "spoiled"):
        folder = raw / label
        folder.mkdir(parents=True)
        for index in range(10):
            make_image(folder / f"{index}.jpg")

    summary = prepare_classification_dataset(
        raw, tmp_path / "processed", DatasetSplit(60, 20, 20), seed=7
    )

    assert summary == {
        "fresh": {"train": 6, "val": 2, "test": 2},
        "spoiled": {"train": 6, "val": 2, "test": 2},
    }
    assert (tmp_path / "processed" / "test" / "spoiled" / "9.jpg").exists() or len(
        list((tmp_path / "processed" / "test" / "spoiled").glob("*.jpg"))
    ) == 2


def test_prepare_dataset_rejects_missing_class(tmp_path: Path):
    (tmp_path / "raw" / "fresh").mkdir(parents=True)

    with pytest.raises(ValueError, match="spoiled"):
        prepare_classification_dataset(tmp_path / "raw", tmp_path / "processed")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fruit_grader.dataset'`.

- [ ] **Step 3: Implement the minimal dataset module**

Implement `DatasetSplit` as a frozen dataclass with `train`, `val`, `test` integer fields and a `validate()` method that rejects totals other than 100. Implement `prepare_classification_dataset` to:

1. Require both labels in `("fresh", "spoiled")`;
2. Use Pillow `Image.verify()` to retain only readable `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files;
3. Seed a local `random.Random(seed)` instance once per label and shuffle the sorted source file list;
4. Compute `train = floor(count * train / 100)`, `val = floor(count * val / 100)`, and place the remainder in `test`;
5. Delete only `output_root` after resolving it and rejecting filesystem roots; recreate the six target directories;
6. Copy source images with `shutil.copy2` and return the count mapping in the test's exact shape.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`

Expected: PASS.

- [ ] **Step 5: Add the data-preparation CLI and test its parser**

Add `build_parser()` and `main()` to `scripts/prepare_dataset.py` with required `--raw-dir` and `--output-dir`, optional `--seed` defaulting to `42`, and optional `--split` defaulting to `70,15,15`. Add this test:

```python
from scripts.prepare_dataset import build_parser


def test_prepare_parser_uses_default_seed():
    args = build_parser().parse_args(["--raw-dir", "raw", "--output-dir", "processed"])

    assert args.seed == 42
    assert args.split == "70,15,15"
```

Run: `pytest tests/test_cli.py::test_prepare_parser_uses_default_seed -v`

Expected before implementation: FAIL with import error. Expected after implementation: PASS.

- [ ] **Step 6: Commit data preparation**

```bash
git add src/fruit_grader/dataset.py scripts/prepare_dataset.py tests/test_dataset.py tests/test_cli.py
git commit -m "feat: prepare reproducible image dataset"
```

## Task 3: Prediction result normalization

**Files:**

- Create: `src/fruit_grader/inference.py`
- Modify: `tests/test_inference.py`

**Interfaces:**

- Produces `Prediction(label: str, confidence: float, message: str)`.
- Produces `prediction_from_values(names: Mapping[int, str], top1: int, top1conf: float) -> Prediction`.
- Produces `prediction_from_result(result: Any) -> Prediction`, which adapts a single Ultralytics classification result.

- [ ] **Step 1: Write failing tests for Chinese messages and unknown labels**

```python
# tests/test_inference.py
from fruit_grader.inference import prediction_from_values


def test_fresh_prediction_uses_chinese_fresh_message():
    prediction = prediction_from_values({0: "fresh", 1: "spoiled"}, 0, 0.875)

    assert prediction.label == "fresh"
    assert prediction.confidence == 0.875
    assert prediction.message == "判别结果：新鲜"


def test_unknown_prediction_is_not_mislabeled():
    prediction = prediction_from_values({0: "unknown"}, 0, 0.5)

    assert prediction.message == "判别结果：未知类别（unknown）"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_inference.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fruit_grader.inference'`.

- [ ] **Step 3: Implement the minimal inference module**

Define `Prediction` as a frozen dataclass. `prediction_from_values` must map only `fresh` to `判别结果：新鲜` and `spoiled` to `判别结果：变质`; any other string must produce `判别结果：未知类别（<label>）`. Convert the confidence to `float`. `prediction_from_result` must read `result.names`, `result.probs.top1`, and `result.probs.top1conf` and delegate to `prediction_from_values`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_inference.py -v`

Expected: PASS.

- [ ] **Step 5: Commit inference normalization**

```bash
git add src/fruit_grader/inference.py tests/test_inference.py
git commit -m "feat: normalize freshness predictions"
```

## Task 4: Train and command-line prediction entry points

**Files:**

- Create: `train.py`
- Create: `predict.py`
- Modify: `tests/test_cli.py`

**Interfaces:**

- Consumes: processed classification data root, image path, and a YOLO `.pt` model path.
- Produces: `build_train_parser()`, `build_predict_parser()`, and executable `main()` functions.

- [ ] **Step 1: Write failing parser tests**

```python
from predict import build_parser as build_predict_parser
from train import build_parser as build_train_parser


def test_train_parser_uses_yolo_classification_default():
    args = build_train_parser().parse_args(["--data", "data/processed"])

    assert args.model == "yolo26n-cls.pt"
    assert args.epochs == 50


def test_predict_parser_accepts_image_and_model_paths():
    args = build_predict_parser().parse_args([
        "--image", "example.jpg", "--model", "models/best.pt"
    ])

    assert args.image == "example.jpg"
    assert args.model == "models/best.pt"
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL with `ModuleNotFoundError` for `train` or `predict`.

- [ ] **Step 3: Implement minimal CLI behavior**

Implement `train.py` to parse `--data`, `--model` (default `yolo26n-cls.pt`), `--epochs` (default `50`), `--imgsz` (default `224`), `--batch` (default `16`), and `--project` (default `runs`). Before importing or creating `YOLO`, verify `--data` contains `train/fresh`, `train/spoiled`, `val/fresh`, and `val/spoiled`; raise a Chinese `FileNotFoundError` otherwise. Then call `YOLO(args.model).train(...)`, load the returned `best.pt`, and call `.val(data=args.data, split="test")` if a test directory exists.

Implement `predict.py` to parse required `--image` and `--model`, verify both files exist, call `YOLO(model_path).predict(source=image_path, verbose=False)`, convert the first result with `prediction_from_result`, and print `message` and percentage confidence with two decimals.

- [ ] **Step 4: Run parser tests to verify they pass**

Run: `pytest tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Run help commands**

Run: `python train.py --help`

Expected: exit code 0 and options including `--data`, `--model`, and `--epochs`.

Run: `python predict.py --help`

Expected: exit code 0 and options including `--image` and `--model`.

- [ ] **Step 6: Commit command-line entry points**

```bash
git add train.py predict.py tests/test_cli.py
git commit -m "feat: add training and prediction commands"
```

## Task 5: Streamlit upload interface

**Files:**

- Create: `app.py`
- Modify: `src/fruit_grader/inference.py`
- Modify: `tests/test_inference.py`

**Interfaces:**

- Produces `validate_image_bytes(image_bytes: bytes) -> None`, raising `ValueError` with a Chinese message for invalid files.
- Produces a Streamlit page accepting `.jpg`, `.jpeg`, and `.png` uploads and a configurable model path.

- [ ] **Step 1: Write failing upload-validation tests**

```python
import pytest
from PIL import Image
from io import BytesIO

from fruit_grader.inference import validate_image_bytes


def test_validate_image_bytes_accepts_png():
    output = BytesIO()
    Image.new("RGB", (8, 8), "green").save(output, format="PNG")

    assert validate_image_bytes(output.getvalue()) is None


def test_validate_image_bytes_rejects_non_image():
    with pytest.raises(ValueError, match="无法读取图片"):
        validate_image_bytes(b"not an image")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_inference.py -v`

Expected: FAIL with `ImportError: cannot import name 'validate_image_bytes'`.

- [ ] **Step 3: Implement validation and the page**

Implement `validate_image_bytes` with `PIL.Image.open(BytesIO(image_bytes)).verify()` and translate `UnidentifiedImageError` and `OSError` into `ValueError("无法读取图片，请上传有效的 JPG 或 PNG 文件。")`.

Implement `app.py` with a Chinese title, a sidebar text field defaulting to `models/best.pt`, an uploader limited to JPG/JPEG/PNG, an original-image preview, a model-file existence check, image-byte validation, lazy YOLO model loading with `@st.cache_resource`, and fields showing the Chinese conclusion, original label and confidence percentage. The page must state that the result is only an external image judgment.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inference.py -v`

Expected: PASS.

- [ ] **Step 5: Perform a Streamlit startup check**

Run: `streamlit run app.py --server.headless true --server.port 8501`

Expected: the process reports a local URL and stays running; stop it after observing the startup message.

- [ ] **Step 6: Commit the upload interface**

```bash
git add app.py src/fruit_grader/inference.py tests/test_inference.py
git commit -m "feat: add image upload interface"
```

## Task 6: User documentation and end-to-end verification

**Files:**

- Create: `README.md`
- Modify: `data/README.md`

**Interfaces:**

- Produces a Chinese quick-start guide covering environment setup, public-data source link, raw-data layout, preparation, training, prediction, app startup, metrics, and limitations.

- [ ] **Step 1: Write the documentation acceptance checklist in README**

The README must have these headings: `项目功能`, `环境安装`, `数据集准备`, `训练模型`, `图片预测`, `启动网页`, `评价指标`, `项目限制`. It must link to the public Mendeley data record `https://data.mendeley.com/datasets/6ps7gtp2wg/1` and explain that the user must place images into the two raw class folders.

- [ ] **Step 2: Write the README and data instructions**

Document these exact commands:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/processed
python train.py --data data/processed
python predict.py --image examples/sample.jpg --model models/best.pt
streamlit run app.py
```

Explain that the `examples/sample.jpg` path is illustrative and must be replaced with an actual image path. State that publicly sourced data must retain its original license and citation information. State that a score from this model is not a food-safety conclusion.

- [ ] **Step 3: Run all automated tests**

Run: `pytest -v`

Expected: all tests PASS with no skipped failures.

- [ ] **Step 4: Run command verification**

Run: `python scripts/prepare_dataset.py --help`

Expected: exit code 0.

Run: `python train.py --help`

Expected: exit code 0.

Run: `python predict.py --help`

Expected: exit code 0.

- [ ] **Step 5: Commit documentation and verified implementation**

```bash
git add README.md data/README.md
git commit -m "docs: add fruit classifier quick start"
```

## Plan Self-Review

- Spec coverage: Tasks 1–6 respectively cover configuration, data preparation, prediction transformation, train/predict commands, webpage upload, documentation, and the full verification requirements.
- Placeholder scan: no `TODO`, `TBD`, or deferred implementation wording remains in tasks.
- Interface consistency: `DatasetSplit`, `prepare_classification_dataset`, `Prediction`, `prediction_from_values`, `prediction_from_result`, and `validate_image_bytes` are defined before their consuming tasks.
