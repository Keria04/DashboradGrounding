@echo off
chcp 65001 > nul
echo ══════════════════════════════════════════════════════════════════
echo 📤 上传项目到GitHub
echo ══════════════════════════════════════════════════════════════════
echo.
echo 请确保已完成以下准备：
echo   1. 已安装Git（git --version可以运行）
echo   2. 已在GitHub创建仓库
echo   3. 已获取仓库地址
echo.
echo ══════════════════════════════════════════════════════════════════
echo.

REM 检查Git是否安装
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git未安装！
    echo.
    echo 请先安装Git:
    echo   访问 https://git-scm.com/download/win
    echo   下载并安装Git for Windows
    echo.
    pause
    exit /b 1
)

echo ✅ Git已安装
echo.

REM 检查是否已初始化
if exist .git (
    echo ✅ Git仓库已初始化
) else (
    echo 📦 初始化Git仓库...
    git init
    echo ✅ 初始化完成
)
echo.

REM 询问GitHub仓库地址
set /p REPO_URL="请输入您的GitHub仓库地址（例如: https://github.com/用户名/Project_srp.git）: "

echo.
echo 📡 设置远程仓库...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
echo ✅ 远程仓库已设置: %REPO_URL%
echo.

REM 配置Git LFS
echo 📦 配置Git LFS（处理大文件）...
git lfs install
if errorlevel 1 (
    echo ⚠️ Git LFS未安装，将跳过
    echo.
    echo 如果模型文件太大无法上传，请安装Git LFS:
    echo   下载: https://git-lfs.github.com/
    echo.
) else (
    git lfs track "*.pt"
    echo ✅ Git LFS已配置
)
echo.

REM 添加文件
echo 📄 添加文件到Git...
git add .
echo ✅ 文件已添加
echo.

REM 显示状态
echo 📊 准备提交的文件:
git status --short
echo.

REM 提交
set /p COMMIT_MSG="请输入提交信息（直接回车使用默认）: "
if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Initial commit: Dashboard Chart Detection System v2.1 (mAP50=51.2%%)
)

echo.
echo 💾 提交到本地仓库...
git commit -m "%COMMIT_MSG%"
echo ✅ 提交完成
echo.

REM 设置主分支
git branch -M main
echo ✅ 分支已设置为main
echo.

echo ══════════════════════════════════════════════════════════════════
echo 📤 准备推送到GitHub
echo ══════════════════════════════════════════════════════════════════
echo.
echo ⚠️ 接下来会要求输入GitHub凭据:
echo.
echo 方式1（推荐）: 使用Personal Access Token
echo   - Username: 你的GitHub用户名
echo   - Password: 粘贴你的Token（不是GitHub密码）
echo.
echo 如何获取Token:
echo   1. GitHub → Settings → Developer settings
echo   2. Personal access tokens → Tokens (classic)
echo   3. Generate new token
echo   4. 勾选 repo 权限
echo   5. 复制token
echo.
echo 方式2: 使用SSH（如果已配置SSH密钥）
echo.
echo ══════════════════════════════════════════════════════════════════
echo.
pause

echo 🚀 开始推送...
echo.
git push -u origin main

if errorlevel 1 (
    echo.
    echo ══════════════════════════════════════════════════════════════════
    echo ❌ 推送失败
    echo ══════════════════════════════════════════════════════════════════
    echo.
    echo 可能的原因:
    echo   1. 凭据错误（用户名或Token不正确）
    echo   2. 仓库地址错误
    echo   3. 网络问题
    echo   4. 文件太大（需要Git LFS）
    echo.
    echo 解决方案:
    echo   - 检查GitHub用户名和Token是否正确
    echo   - 确保仓库地址正确
    echo   - 如果是文件太大，安装Git LFS
    echo.
    pause
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════════════
echo ✅ 上传成功！
echo ══════════════════════════════════════════════════════════════════
echo.
echo 🎉 您的项目已上传到GitHub！
echo.
echo 📌 仓库地址: %REPO_URL%
echo.
echo 🌐 访问您的项目:
echo    在浏览器中打开上面的地址（去掉.git后缀）
echo.
echo 📋 下一步:
echo    1. 访问GitHub仓库查看项目
echo    2. 完善README（添加截图、演示链接等）
echo    3. 添加License（Settings → Add license）
echo    4. 添加Topics标签（yolo, gradio, object-detection等）
echo    5. 分享仓库链接给其他人
echo.
echo ══════════════════════════════════════════════════════════════════
pause

