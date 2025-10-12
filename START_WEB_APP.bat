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
echo 🌐 启动仪表盘图表检测 - Web界面
echo ======================================================================
echo.
echo 功能特性:
echo   ✓ 友好的浏览器界面
echo   ✓ 拖拽上传图片
echo   ✓ 实时预览检测结果
echo   ✓ 可调节检测参数
echo   ✓ 导出详细JSON结果
echo   ⚡ GPU加速推理
echo.
echo 正在启动Web服务器...
echo 浏览器将自动打开，如果没有请手动访问: http://127.0.0.1:7860
echo.
echo 按 Ctrl+C 停止服务器
echo ======================================================================
echo.

python scripts/web_app.py

pause

