"""Train a YOLO image classifier for fruit freshness."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练水果新鲜度 YOLO 分类模型。")
    parser.add_argument("--data", required=True, help="包含 train、val、test 的分类数据集目录")
    parser.add_argument("--model", default="yolo26n-cls.pt", help="YOLO 分类预训练权重")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=224, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="训练批大小")
    parser.add_argument("--project", default="runs", help="Ultralytics 训练输出目录")
    parser.add_argument("--name", default="fruit_freshness", help="本次训练名称")
    parser.add_argument("--output-model", default="models/best.pt", help="保存最佳权重的位置")
    return parser


def validate_dataset_root(data_root: Path) -> None:
    required_dirs = [
        data_root / split / label
        for split in ("train", "val")
        for label in ("fresh", "spoiled")
    ]
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("数据集目录不完整，缺少：" + "，".join(missing))


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data)
    validate_dataset_root(data_root)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    best_path = Path(model.trainer.best)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output_model)
    print(f"最佳模型已保存到：{output_model}")

    if (data_root / "test").is_dir():
        YOLO(str(output_model)).val(data=str(data_root), split="test")


if __name__ == "__main__":
    main()
