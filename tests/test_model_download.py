from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.download_model import (
    download_model,
    main,
    sha256_file,
    verify_model_file,
)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_sha256_file_returns_expected_digest(tmp_path: Path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"model-bytes")

    assert (
        sha256_file(model)
        == "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5"
    )


def test_verify_model_file_accepts_matching_sha256(tmp_path: Path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"model-bytes")

    assert (
        verify_model_file(
            model,
            "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
        )
        is None
    )


def test_verify_model_file_rejects_wrong_sha256(tmp_path: Path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"model-bytes")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_model_file(model, "0" * 64)


def test_download_model_rejects_non_https_url(tmp_path: Path):
    destination = tmp_path / "best.pt"
    destination.write_bytes(b"existing")

    with pytest.raises(ValueError, match="HTTPS"):
        download_model(
            "http://example.com/best.pt",
            "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
            destination,
        )

    assert destination.read_bytes() == b"existing"
    assert not destination.with_suffix(".pt.part").exists()


def test_download_model_replaces_destination_after_checksum_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "best.pt"
    destination.write_bytes(b"existing")
    payload = b"model-bytes"

    monkeypatch.setattr(
        "scripts.download_model.urlopen",
        lambda url: FakeResponse(payload),
    )

    result = download_model(
        "https://example.com/best.pt",
        "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
        destination,
    )

    assert result == destination
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".pt.part").exists()


def test_download_model_preserves_existing_destination_on_failed_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "best.pt"
    destination.write_bytes(b"existing")

    monkeypatch.setattr(
        "scripts.download_model.urlopen",
        lambda url: FakeResponse(b"tampered"),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        download_model(
            "https://example.com/best.pt",
            "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
            destination,
        )

    assert destination.read_bytes() == b"existing"
    assert not destination.with_suffix(".pt.part").exists()


def test_main_rejects_manifest_missing_sha256_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "v1.0.0",
                "filename": "best.pt",
                "url": "https://example.com/best.pt",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "download_model.py",
            "--manifest",
            str(manifest),
            "--destination",
            str(tmp_path / "best.pt"),
        ],
    )

    with pytest.raises(ValueError, match="sha256"):
        main()


def test_main_rejects_manifest_when_url_is_not_a_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "v1.0.0",
                "filename": "best.pt",
                "url": 123,
                "sha256": "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "download_model.py",
            "--manifest",
            str(manifest),
            "--destination",
            str(tmp_path / "best.pt"),
        ],
    )

    with pytest.raises(ValueError, match="url"):
        main()


def test_main_reads_manifest_and_passes_arguments_to_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = tmp_path / "model-manifest.json"
    destination = tmp_path / "best.pt"
    manifest.write_text(
        json.dumps(
            {
                "version": "v1.0.0",
                "filename": "best.pt",
                "url": "https://example.com/best.pt",
                "sha256": "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_download_model(url: str, expected_sha256: str, target: Path) -> Path:
        captured["url"] = url
        captured["sha256"] = expected_sha256
        captured["destination"] = target
        return target

    monkeypatch.setattr("scripts.download_model.download_model", fake_download_model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_model.py",
            "--manifest",
            str(manifest),
            "--destination",
            str(destination),
        ],
    )

    main()

    assert captured == {
        "url": "https://example.com/best.pt",
        "sha256": "357e5d6fafa34d27360fec24b4326d3534905e33c6acdee60198fb078b7b79e5",
        "destination": destination,
    }
