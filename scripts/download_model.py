from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_file(path: Path, expected_sha256: str) -> None:
    actual_sha256 = sha256_file(path).lower()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("模型 SHA-256 校验失败。")


def download_model(url: str, expected_sha256: str, destination: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError("模型下载地址必须使用 HTTPS。")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".pt.part")

    try:
        with urlopen(url) as response, temp_path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        verify_model_file(temp_path, expected_sha256)
        temp_path.replace(destination)
        return destination
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_manifest(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required_fields = ("filename", "url", "sha256")

    for field in required_fields:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest 字段 {field} 必须是非空字符串。")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载并校验 fruit freshness 模型文件。")
    parser.add_argument("--manifest", required=True, help="模型清单 JSON 文件路径。")
    parser.add_argument("--destination", required=True, help="模型输出路径。")
    return parser


def main() -> Path:
    args = build_parser().parse_args()
    manifest = load_manifest(Path(args.manifest))
    return download_model(
        manifest["url"],
        manifest["sha256"],
        Path(args.destination),
    )


if __name__ == "__main__":
    main()
