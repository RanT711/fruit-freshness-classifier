"""Run fruit-freshness classification for one local image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fruit_grader.inference import prediction_from_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="预测单张水果图片的新鲜度。")
    parser.add_argument("--image", required=True, help="待预测图片路径")
    parser.add_argument("--model", required=True, help="训练完成的 YOLO .pt 模型路径")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    from ultralytics import YOLO

    results = YOLO(str(model_path)).predict(source=str(image_path), verbose=False)
    if not results:
        raise RuntimeError("YOLO 未返回预测结果。")

    prediction = prediction_from_result(results[0])
    print(prediction.message)
    print(f"类别标签：{prediction.label}")
    print(f"置信度：{prediction.confidence:.2%}")


if __name__ == "__main__":
    main()
