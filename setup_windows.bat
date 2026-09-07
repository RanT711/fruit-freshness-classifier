@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "PYTHONUTF8=1"

pushd "%~dp0"
if errorlevel 1 (
    echo Cannot enter project directory.
    exit /b 1
)

echo [1/4] Checking Python 3...
py -3 -V >nul 2>&1
if errorlevel 1 (
    echo Python 3 with the Windows launcher was not found.
    popd
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        popd
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment already exists.
)

echo [3/4] Installing runtime dependencies...
.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
if errorlevel 1 (
    echo Dependency installation failed.
    popd
    exit /b 1
)

if not exist "models\best.pt" (
    echo [4/4] Bundled model is missing: models\best.pt
    echo Please download the repository again or restore the model file.
    popd
    exit /b 1
)

echo [4/4] Bundled model found: models\best.pt
echo Setup complete. Run run_web.bat to start the app.
popd
exit /b 0
