@echo off
chcp 65001 > nul
echo ══════════════════════════════════════════════════════════════════
echo 📤 上传到团队GitHub仓库
echo ══════════════════════════════════════════════════════════════════
echo.
echo 目标仓库: https://github.com/Keria04/DashboradGrounding
echo.
echo ══════════════════════════════════════════════════════════════════
echo.

REM 检查Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git未安装！请先安装Git
    pause
    exit /b 1
)
echo ✅ Git已安装
echo.

REM 初始化Git（如果需要）
if not exist .git (
    echo 📦 初始化Git仓库...
    git init
    echo ✅ 初始化完成
) else (
    echo ✅ Git仓库已存在
)
echo.

REM 设置远程仓库
echo 📡 设置远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/Keria04/DashboradGrounding.git
echo ✅ 远程仓库: https://github.com/Keria04/DashboradGrounding
echo.

REM 配置Git LFS
echo 📦 配置Git LFS...
git lfs install
git lfs track "*.pt"
git add .gitattributes
echo ✅ Git LFS已配置
echo.

REM 拉取远程代码（合并现有内容）
echo 📥 拉取远程仓库内容...
git pull origin main --allow-unrelated-histories --no-edit 2>nul
if errorlevel 1 (
    echo ⚠️ 远程仓库可能为空或首次推送，继续...
)
echo.

REM 添加文件
echo 📄 添加文件...
git add .
echo ✅ 文件已添加
echo.

REM 显示状态
echo 📊 准备提交的文件:
git status --short
echo.

REM 提交
echo 💾 提交到本地仓库...
git commit -m "添加Phase 1完整代码: YOLOv8s检测系统 (mAP50=51.2%%)"
if errorlevel 1 (
    echo ⚠️ 没有新的更改需要提交
)
echo.

REM 设置分支
git branch -M main
echo.

echo ══════════════════════════════════════════════════════════════════
echo 📤 准备推送到GitHub
echo ══════════════════════════════════════════════════════════════════
echo.
echo ⚠️ 接下来需要输入GitHub凭据
echo.
echo 请准备:
echo   Username: 你的GitHub用户名（例如: Keria04）
echo   Password: Personal Access Token（不是密码！）
echo.
echo 如何获取Token:
echo   1. GitHub → Settings → Developer settings
echo   2. Personal access tokens → Tokens (classic)
echo   3. Generate new token
echo   4. 勾选 "repo" 权限
echo   5. 复制token
echo.
pause

echo.
echo 🚀 开始推送...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ══════════════════════════════════════════════════════════════════
    echo ❌ 推送失败
    echo ══════════════════════════════════════════════════════════════════
    echo.
    echo 可能的原因:
    echo   1. 没有该仓库的写权限
    echo   2. 凭据错误
    echo   3. 网络问题
    echo.
    echo 解决方案:
    echo   - 确认您已被添加为仓库的Collaborator
    echo   - 检查Token权限是否包含repo
    echo   - 重试推送: git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════════════
echo ✅ 上传成功！
echo ══════════════════════════════════════════════════════════════════
echo.
echo 🎉 您的代码已上传到团队仓库！
echo.
echo 📌 仓库地址: https://github.com/Keria04/DashboradGrounding
echo.
echo 🌐 访问查看:
echo    在浏览器打开上面的链接
echo.
echo 📋 下一步:
echo    1. 访问仓库验证代码已上传
echo    2. 创建Pull Request（如果需要review）
echo    3. 通知团队成员查看
echo.
echo ══════════════════════════════════════════════════════════════════
pause

