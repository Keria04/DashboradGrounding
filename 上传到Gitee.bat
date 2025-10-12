@echo off
chcp 65001 > nul
echo ══════════════════════════════════════════════════════════════════
echo 📤 上传到Gitee（码云）- 国内平台
echo ══════════════════════════════════════════════════════════════════
echo.
echo Gitee优势:
echo   ✅ 国内访问快速稳定
echo   ✅ 免费LFS 5GB
echo   ✅ 可自动同步到GitHub
echo   ✅ 使用密码，不需要Token
echo.
echo ══════════════════════════════════════════════════════════════════
echo.

REM 检查Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git未安装！
    pause
    exit /b 1
)

echo ✅ Git已安装
echo.

REM 询问Gitee仓库地址
echo 请先在 https://gitee.com/ 创建仓库
echo.
set /p GITEE_URL="输入您的Gitee仓库地址（例如: https://gitee.com/用户名/dashboard-detection.git）: "

if "%GITEE_URL%"=="" (
    echo ❌ 未输入仓库地址
    pause
    exit /b 1
)

echo.
echo 📡 设置Gitee远程仓库...
git remote remove origin 2>nul
git remote add origin %GITEE_URL%
echo ✅ 远程仓库: %GITEE_URL%
echo.

REM 配置Git LFS
echo 📦 配置Git LFS...
git lfs install
git lfs track "*.pt"
git add .gitattributes
echo ✅ Git LFS已配置
echo.

REM 确保代码已提交
git add .
git commit -m "Phase 1: YOLOv8s检测系统 (mAP50=51.2%%)" 2>nul

REM 设置分支
git branch -M main
echo.

echo ══════════════════════════════════════════════════════════════════
echo 📤 开始推送到Gitee
echo ══════════════════════════════════════════════════════════════════
echo.
echo 💡 提示: Gitee使用登录密码，不需要Token
echo.
echo 即将要求输入:
echo   Username: 你的Gitee用户名
echo   Password: 你的Gitee登录密码
echo.
pause

echo.
echo 🚀 推送中...
echo.
git push -u origin main

if errorlevel 1 (
    echo.
    echo ══════════════════════════════════════════════════════════════════
    echo ❌ 推送失败
    echo ══════════════════════════════════════════════════════════════════
    echo.
    echo 可能的原因:
    echo   1. 用户名或密码错误
    echo   2. 仓库地址错误
    echo   3. 网络问题
    echo.
    echo 解决方案:
    echo   - 检查Gitee用户名和密码
    echo   - 确认仓库地址正确
    echo   - 重试: git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════════════
echo ✅ 上传成功！
echo ══════════════════════════════════════════════════════════════════
echo.
echo 🎉 代码已上传到Gitee！
echo.
echo 📌 Gitee仓库: %GITEE_URL%
echo.
echo 🌐 访问查看:
echo    浏览器打开上面的地址（去掉.git）
echo.
echo 📋 下一步（可选）:
echo    1. 在Gitee配置同步到GitHub
echo       管理 → 仓库镜像管理 → 添加镜像
echo       目标: https://github.com/Keria04/DashboradGrounding.git
echo.
echo    2. 分享Gitee链接给团队
echo.
echo    3. 以后推送: git push（自动同步到Gitee和GitHub）
echo.
echo ══════════════════════════════════════════════════════════════════
pause

