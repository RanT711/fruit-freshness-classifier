@echo off
setlocal
set "PYTHONUTF8=1"

pushd "%~dp0"
if errorlevel 1 (
    echo 无法进入项目目录。
    set "STATUS=%ERRORLEVEL%"
    exit /b %STATUS%
)

echo [1/4] 检查 Python 3 环境...
py -3 -V >nul 2>&1
if errorlevel 1 (
    echo 未找到 py -3，请先安装 Python 3 并启用 Windows 启动器。
    set "STATUS=%ERRORLEVEL%"
    popd
    exit /b %STATUS%
)

if not exist ".venv\\Scripts\\python.exe" (
    echo [2/4] 创建虚拟环境...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo 创建虚拟环境失败。
        set "STATUS=%ERRORLEVEL%"
        popd
        exit /b %STATUS%
    )
) else (
    echo [2/4] 已检测到虚拟环境，跳过创建。
)

echo [3/4] 安装运行依赖...
.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
if errorlevel 1 (
    echo 安装依赖失败。
    set "STATUS=%ERRORLEVEL%"
    popd
    exit /b %STATUS%
)

if not exist "models" (
    mkdir "models"
    if errorlevel 1 (
        echo 创建 models 目录失败。
        set "STATUS=%ERRORLEVEL%"
        popd
        exit /b %STATUS%
    )
)

echo [4/4] 下载并校验模型文件...
.venv\Scripts\python.exe scripts\download_model.py --manifest model-manifest.json --destination models\best.pt
if errorlevel 1 (
    echo 下载或校验模型失败。
    set "STATUS=%ERRORLEVEL%"
    popd
    exit /b %STATUS%
)

echo 安装完成，可以运行 run_web.bat 启动页面。
popd
exit /b 0
