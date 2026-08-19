"""Create a YOLO image-classification directory from raw freshness images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fruit_grader.dataset import DatasetSplit, prepare_classification_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="整理水果新鲜度图片数据集。")
    parser.add_argument("--raw-dir", required=True, help="包含 fresh 与 spoiled 的原始图片目录")
    parser.add_argument("--output-dir", required=True, help="生成 train、val、test 的输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机切分种子")
    parser.add_argument("--split", default="70,15,15", help="训练、验证、测试比例，例如 70,15,15")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parts = [part.strip() for part in args.split.split(",")]
    if len(parts) != 3:
        raise ValueError("--split 必须包含三个逗号分隔的整数，例如 70,15,15。")

    try:
        split = DatasetSplit(*(int(part) for part in parts))
    except ValueError as error:
        raise ValueError("--split 必须包含三个整数，例如 70,15,15。") from error

    summary = prepare_classification_dataset(
        Path(args.raw_dir), Path(args.output_dir), split=split, seed=args.seed
    )
    for label, counts in summary.items():
        print(f"{label}: train={counts['train']}, val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()
