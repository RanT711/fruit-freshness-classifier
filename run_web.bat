@echo off
setlocal
set "PYTHONUTF8=1"

pushd "%~dp0"
if errorlevel 1 (
    echo 无法进入项目目录。
    set "STATUS=%ERRORLEVEL%"
    exit /b %STATUS%
)

if not exist ".venv\\Scripts\\python.exe" (
    echo 未找到 .venv\\Scripts\\python.exe，请先运行 setup_windows.bat。
    popd
    exit /b 1
)

if not exist "models\\best.pt" (
    echo 未找到 models\\best.pt，请先运行 setup_windows.bat 下载模型。
    popd
    exit /b 1
)

echo 正在启动本地识别页面...
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
if errorlevel 1 (
    echo 启动页面失败。
    set "STATUS=%ERRORLEVEL%"
    popd
    exit /b %STATUS%
)

popd
exit /b 0
