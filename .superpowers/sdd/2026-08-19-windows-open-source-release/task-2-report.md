# Task 2 Report

## Status

Completed on August 19, 2026.

## Scope delivered

- Added `scripts/download_model.py` as a standalone CLI with `--manifest` and `--destination`.
- Added `model-manifest.json` with the exact `v1.0.0` release URL and SHA-256 from the task brief.
- Added focused downloader tests in `tests/test_model_download.py`.

## TDD evidence

1. Wrote `tests/test_model_download.py` before any production downloader code existed.
2. Ran `python -m pytest tests/test_model_download.py -q`.
3. Observed the expected red state: `ModuleNotFoundError: No module named 'scripts.download_model'`.
4. Implemented the smallest downloader needed to satisfy the tests.
5. Re-ran focused tests to green, then re-ran the full suite.

## Behaviors verified

- Computes SHA-256 digests from files.
- Accepts matching model digests and rejects mismatches.
- Rejects non-HTTPS model URLs.
- Downloads into a temporary `.pt.part` file.
- Verifies the temporary file before replacing the destination.
- Preserves any pre-existing destination when verification fails.
- Validates manifest `filename`, `url`, and `sha256` fields as non-empty strings.
- Passes manifest values through the CLI entrypoint into the downloader.

## Self-review

- Kept the implementation limited to the task scope: no UI, docs, requirements, or Windows script changes.
- Used a focused fake response object in tests instead of a real remote, so network behavior is covered without creating a release.
- Confirmed cleanup behavior removes only the temporary `.pt.part` file on failure.

## Verification

- `python -m pytest tests/test_model_download.py -q` → `9 passed`
- `python -m pytest -q` → `28 passed`

## Commit

- Intended commit message: `feat: verify model downloads`
