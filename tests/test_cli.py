from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_can_be_imported():
    import fruit_grader

    assert fruit_grader.__name__ == "fruit_grader"


def test_prepare_parser_uses_default_seed():
    from scripts.prepare_dataset import build_parser

    args = build_parser().parse_args(["--raw-dir", "raw", "--output-dir", "processed"])

    assert args.seed == 42
    assert args.split == "70,15,15"


def test_train_parser_uses_yolo_classification_default():
    from train import build_parser

    args = build_parser().parse_args(["--data", "data/processed"])

    assert args.model == "yolo26n-cls.pt"
    assert args.epochs == 50


def test_predict_parser_accepts_image_and_model_paths():
    from predict import build_parser

    args = build_parser().parse_args(
        ["--image", "example.jpg", "--model", "models/best.pt"]
    )

    assert args.image == "example.jpg"
    assert args.model == "models/best.pt"


def test_predict_help_runs_as_direct_script():
    result = subprocess.run(
        [sys.executable, "predict.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout


def test_prepare_dataset_help_runs_as_direct_script():
    result = subprocess.run(
        [sys.executable, "scripts/prepare_dataset.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--raw-dir" in result.stdout
