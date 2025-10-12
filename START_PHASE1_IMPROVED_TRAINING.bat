@echo off
chcp 65001 > nul

REM 激活虚拟环境
echo 激活虚拟环境...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败！
    pause
    exit /b 1
)

echo ✅ 虚拟环境已激活
echo.
echo ======================================================================
echo 🚀 Phase 1 改进训练 - 基于GPU训练结果优化
echo ======================================================================
echo.
echo 📊 当前性能 (96轮训练结果):
echo    mAP50: 48.3%% (vs 基础37.2%%, +30%%)
echo    Recall: 51.8%% (vs 基础29.5%%, +76%%)
echo    训练状态: 未完成 (96/200轮)
echo.
echo 🎯 Phase 1 改进策略:
echo    ✅ 从best.pt (Epoch 66) 恢复训练
echo    ✅ 启用多尺度训练 (800分辨率)
echo    ✅ 降低学习率 (0.001 → 0.0005)
echo    ✅ 增加patience (30 → 50)
echo    ✅ 降低置信度阈值 (0.10 → 0.05)
echo    ✅ 优化数据增强强度
echo.
echo 🎯 预期目标:
echo    - mAP50: 48.3%% → 52-55%%
echo    - Recall: 51.8%% → 60-65%%
echo    - Line chart: ~25%% → 35-40%%
echo    - Heatmap: ~15%% → 20-25%%
echo.
echo 预计训练时间: 20-25分钟 (GPU)
echo ======================================================================
echo.

REM 使用Ultralytics官方训练（正确的命令）
yolo train ^
    data=data/yolo_format/dashboard.yaml ^
    model=experiments/yolov8s_optimized/weights/best.pt ^
    epochs=150 ^
    batch=16 ^
    imgsz=800 ^
    device=0 ^
    workers=8 ^
    patience=50 ^
    lr0=0.0005 ^
    lrf=0.001 ^
    cos_lr=True ^
    hsv_h=0.03 ^
    hsv_s=0.7 ^
    hsv_v=0.4 ^
    erasing=0.2 ^
    mixup=0.10 ^
    conf=0.05 ^
    iou=0.35 ^
    project=experiments ^
    name=yolov8s_phase1_improved ^
    exist_ok=True ^
    amp=True ^
    plots=True ^
    verbose=True

echo.
echo ======================================================================
echo 训练已结束
echo ======================================================================
echo.
echo 📊 查看结果:
echo    结果目录: experiments\yolov8s_phase1_improved\
echo    最佳模型: weights\best.pt
echo    训练曲线: results.png
echo    混淆矩阵: confusion_matrix.png
echo.
echo 📌 下一步:
echo    1. 测试新模型: START_WEB_APP.bat
echo    2. 查看详细分析: GPU训练分析报告_YOLOv8s.md
echo    3. 如果效果好，继续Phase 2(增加数据)
echo.
pause

