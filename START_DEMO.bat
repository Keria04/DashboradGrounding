@echo off
chcp 65001 > nul

REM 激活虚拟环境
echo 激活虚拟环境...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败！
    echo 请确保 .venv 文件夹存在
    pause
    exit /b 1
)

echo ✅ 虚拟环境已激活
echo.
echo ======================================================================
echo 💻 仪表盘图表检测 - 命令行Demo (GPU加速)
echo ======================================================================
echo.
echo 使用方法:
echo   1. 直接运行本脚本 - 检测默认示例图片
echo   2. 拖拽图片到本窗口 - 检测指定图片
echo.
echo ======================================================================
echo.

if "%~1"=="" (
    echo 使用默认示例图片...
    python scripts/app_demo.py --image data/raw/dashboard_0001.png --show
) else (
    echo 检测图片: %~1
    python scripts/app_demo.py --image "%~1" --show
)

echo.
echo ======================================================================
pause

