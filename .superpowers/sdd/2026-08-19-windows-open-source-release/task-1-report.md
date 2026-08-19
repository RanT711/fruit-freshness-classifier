Status: completed

Task brief followed:
- Modified only `app.py`, `src/fruit_grader/inference.py`, and `tests/test_inference.py` for production/test code.
- Kept the existing image-byte-to-array conversion baseline intact.
- Did not modify docs, release scripts, dependencies, or training code.

What changed:
- Added `MAX_UPLOAD_BYTES = 10 * 1024 * 1024` and `MAX_IMAGE_PIXELS = 20_000_000`.
- Added `validate_image_dimensions(width, height, max_pixels=20_000_000)`.
- Added `resolve_trusted_model_path(model_value, models_root)` to constrain model selection to `.pt` files under the trusted `models` directory.
- Updated `validate_image_bytes(image_bytes)` to reject payloads above 10MB and images above 20,000,000 pixels before image verification.
- Updated `app.py` to resolve model paths through the trusted-model validator and surface `ValueError` messages in the UI.

TDD evidence:
- Added failing tests first in `tests/test_inference.py` for oversized dimensions and model paths outside `models`.
- Verified RED with `python -m pytest tests/test_inference.py -q`, which failed during import because `resolve_trusted_model_path` did not yet exist.
- Implemented the minimum code to satisfy the task brief.
- Verified GREEN with `python -m pytest tests/test_inference.py -q`.

Verification evidence:
- Focused tests: `python -m pytest tests/test_inference.py -q` → `7 passed`
- Full suite: `python -m pytest -q` → `17 passed`

Self-review:
- Confirmed only the intended three code files changed for Task 1.
- Confirmed the Streamlit app now rejects model paths outside `models` before loading weights.
- Confirmed missing model files still produce the existing user-facing error after trusted-path validation.

Concerns:
- No dedicated Streamlit UI test coverage was added in this task; behavior is covered through inference-layer tests plus manual path-handling review in `app.py`.
