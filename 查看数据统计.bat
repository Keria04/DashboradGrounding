@echo off
chcp 65001 >nul
echo ======================================================================
echo 📊 数据集统计分析
echo ======================================================================
echo.

REM 激活虚拟环境（如果存在）
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
    echo.
)

REM 运行统计脚本
python scripts\show_data_stats.py --detailed

echo.
echo ======================================================================
echo 按任意键退出...
pause >nul

