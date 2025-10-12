---
title: Dashboard Chart Detector
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🎯 仪表盘图表检测系统

基于YOLOv8s的AI图表自动检测与分类系统。

## 功能特性

- 🚀 自动检测仪表盘中的各种图表区域
- 🎯 识别并分类20种图表类型
- 📊 精确框出每个图表的边界
- ⚡ GPU加速推理

## 模型性能

- **整体准确率**: mAP50 = 51.2%
- **召回率**: 55.7%
- **优势类别**: Map (95%+), Bar chart (85%+), Bubble chart (99%+)

## 支持的图表类型

Map, Bar chart, Line chart, Pie chart, Area chart, Scatter plot, 
Heatmap, Card, Data table, Donut chart, Bubble chart, Radar chart, 
Percentage indicator, Timeline等20种

## 使用提示

- **检测效果优秀的类别**: 使用默认置信度0.10
- **检测效果一般的类别** (Line chart, Heatmap): 降低置信度到0.05
- **误检较多**: 提高置信度到0.20-0.25

## 技术细节

- **模型**: YOLOv8s (11M参数)
- **训练数据**: 144张仪表盘图像
- **训练设备**: NVIDIA GPU
- **框架**: Ultralytics + Gradio

## License

MIT

