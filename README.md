# 🎯 DashBoard Visual Grounding

基于Ultralytics YOLOv8的AI图表自动检测与分类系统。

## ✨ 功能特性

- 🚀 **自动检测**：识别仪表盘中的各种图表区域
- 🎯 **类型分类**：支持20种图表类型（柱状图、折线图、饼图等）
- 🌐 **Web界面**：友好的浏览器操作界面
- ⚡ **GPU加速**：快速推理（支持CPU fallback）

## 📊 模型性能

- **整体准确率**: mAP50 = 51.2%
- **召回率**: 55.7%
- **优势类别**: Map (95%+), Bar chart (80%+), Bubble chart (99%+)

## 🚀 Quick Start

### 1. 下载模型文件

**下载训练好的模型** (必需)：

📦 **模型文件**: `best.pt` (21.5MB)  
🔗 **百度网盘**: https://pan.baidu.com/s/1oVypGjXYjPzEvNgf2vtTzA  
🔑 **提取码**: `srp1`

下载后放置到：`experiments/yolov8s_phase1_improved/weights/best.pt`

> 💡 提示：如果目录不存在，请先创建对应文件夹

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**GPU版本**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics gradio opencv-python numpy pillow pyyaml
```

### 3. 启动Web界面

**Windows**：
```bash
START_WEB_APP.bat
```

**Mac/Linux**：
```bash
python scripts/web_app.py
```

浏览器会自动打开 `http://127.0.0.1:7860`

### 4. 开始检测

1. 拖拽仪表盘图片到上传区
2. 调整参数（可选）
3. 点击"开始检测"
4. 查看结果

## 📁 项目结构

```
Project_srp/
├── scripts/
│   ├── web_app.py              # Web界面主程序
│   ├── app_demo.py             # 命令行检测工具
│   ├── test_ultralytics.py     # 模型测试
│   └── convert_to_yolo_format.py  # 数据转换
│
├── experiments/
│   └── yolov8s_phase1_improved/  # 最新训练结果
│       └── weights/
│           └── best.pt         # ⬇️需从网盘下载 (21.5MB)
│
├── data/
│   ├── annotations/            # 标注文件
│   ├── raw/                    # 原始图像(279张)
│   └── yolo_format/            # YOLO格式数据
│
├── configs/
│   ├── config_yolo.yaml        # 基础配置
│   └── config_yolo_phase1_improved.yaml  # 改进配置
│
├── START_WEB_APP.bat           # 一键启动Web界面
├── CHECK_GPU.bat               # GPU环境检查
└── README.md                   # 本文件
```

## 🎯 支持的图表类型

| 类别 | 检测效果 |
|------|---------|
| Map (地图) | ⭐⭐⭐⭐⭐ 优秀 |
| Bar chart (柱状图) | ⭐⭐⭐⭐⭐ 优秀 |
| Bubble chart (气泡图) | ⭐⭐⭐⭐⭐ 优秀 |
| Card (卡片) | ⭐⭐⭐⭐ 良好 |
| Donut chart (环形图) | ⭐⭐⭐⭐ 良好 |
| Data table (数据表) | ⭐⭐⭐ 中等 |
| Area chart (面积图) | ⭐⭐⭐ 中等 |
| Scatter plot (散点图) | ⭐⭐⭐ 中等 |
| Line chart (折线图) | ⭐⭐ 一般 |
| Heatmap (热力图) | ⭐⭐ 一般 |
| Pie/Radar/Timeline等 | ⭐ 待改进 |


## 🔧 命令行使用

```bash
# 检测单张图片
python scripts/app_demo.py --image data/raw/dashboard_0001.png

# 批量处理
python scripts/app_demo.py --image data/raw --output results

# 调整参数
python scripts/app_demo.py --image image.png --conf 0.05 --iou 0.35
```

## ⚙️ 参数说明

- `--conf`: 置信度阈值（默认0.10，降低可提高召回率）
- `--iou`: NMS IoU阈值（默认0.4）
- `--model`: 指定模型路径（默认自动查找最新）

## 📈 性能优化建议

当前模型性能：51.2% mAP50

**如需提升至60%+**：
1. 收集更多训练数据（重点：Line chart, Heatmap）
2. 使用更大模型（YOLOv8m）
3. 详见项目文档


## 📄  License

MIT License

---

**版本**: v2.1 (Phase 1 改进版)  
**模型性能**: mAP50 = 51.2%  
**更新日期**: 2025-10-12
