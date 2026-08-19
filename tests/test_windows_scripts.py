from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script_name", ["setup_windows.bat", "run_web.bat"])
def test_windows_launcher_scripts_avoid_stale_status_expansion(script_name: str):
    script_text = (PROJECT_ROOT / script_name).read_text(encoding="utf-8")
    normalized = script_text.lower()

    assert "%status%" not in normalized
    assert "setlocal enableextensions enabledelayedexpansion" in normalized or (
        "if errorlevel 1 goto :" in normalized
    )
