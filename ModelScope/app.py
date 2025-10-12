#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仪表盘图表检测系统 - Hugging Face Spaces部署版本
基于YOLOv8s，支持20种图表类型检测
"""

import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import json
from datetime import datetime

class DashboardDetector:
    """仪表盘图表检测器"""
    
    # 20种颜色用于不同类别
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
        (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128),
        (255, 128, 255), (128, 255, 255), (192, 192, 192), (255, 165, 0),
    ]
    
    def __init__(self):
        """初始化检测器"""
        print("📦 正在加载模型...")
        try:
            # 模型文件应该在同一目录下
            self.model = YOLO("best.pt")
            self.class_names = self.model.names
            print(f"✅ 模型加载成功！支持 {len(self.class_names)} 种图表类型")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
            self.class_names = None
    
    def detect(self, image, conf_threshold=0.10, iou_threshold=0.4):
        """
        检测图表
        
        Args:
            image: numpy数组格式的图像
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            
        Returns:
            annotated_image: 标注后的图像
            summary: 检测摘要文本
            detail_json: 详细结果JSON
        """
        if self.model is None:
            return None, "❌ 模型未加载", "{}"
        
        if image is None:
            return None, "❌ 请先上传图像", "{}"
        
        try:
            # 执行检测
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                device='cpu',  # Hugging Face免费版使用CPU
                verbose=False
            )[0]
            
            # 标注图像
            annotated = self._annotate_image(image.copy(), results)
            
            # 生成摘要
            summary = self._generate_summary(results)
            
            # 生成JSON
            detail = self._generate_json(results)
            
            return annotated, summary, detail
            
        except Exception as e:
            return None, f"❌ 检测失败: {str(e)}", "{}"
    
    def _annotate_image(self, image, results):
        """在图像上绘制检测框"""
        boxes = results.boxes
        
        if len(boxes) == 0:
            # 无检测结果
            h, w = image.shape[:2]
            cv2.putText(
                image, "No charts detected!",
                (w//2-150, h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2
            )
            return image
        
        # 绘制每个检测框
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = self.class_names[cls_id]
            
            color = self.COLORS[cls_id % len(self.COLORS)]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # 准备标签
            label = f"#{i+1} {class_name}: {conf:.1%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(
                image,
                (x1, y1-label_h-baseline-8),
                (x1+label_w+10, y1),
                color, -1
            )
            
            # 绘制标签文字
            cv2.putText(
                image, label,
                (x1+5, y1-baseline-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
        
        return image
    
    def _generate_summary(self, results):
        """生成检测摘要"""
        boxes = results.boxes
        
        if len(boxes) == 0:
            return "❌ 未检测到任何图表\n\n💡 **建议**:\n- 降低置信度阈值到0.05\n- 确保图像包含支持的图表类型"
        
        summary = f"✅ **检测到 {len(boxes)} 个图表**:\n\n"
        
        # 统计各类别
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[cls_id]
            conf = float(box.conf[0].cpu().numpy())
            
            if class_name not in class_counts:
                class_counts[class_name] = {'count': 0, 'confs': []}
            class_counts[class_name]['count'] += 1
            class_counts[class_name]['confs'].append(conf)
        
        # 生成列表
        for i, (class_name, info) in enumerate(sorted(
            class_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ), 1):
            avg_conf = np.mean(info['confs'])
            summary += f"{i}. **{class_name}**: {info['count']}个 (平均置信度: {avg_conf:.1%})\n"
        
        return summary
    
    def _generate_json(self, results):
        """生成详细JSON结果"""
        boxes = results.boxes
        detections = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            detections.append({
                'id': i + 1,
                'class': self.class_names[cls_id],
                'class_id': cls_id,
                'confidence': round(conf, 4),
                'bbox': {
                    'x1': round(x1, 2),
                    'y1': round(y1, 2),
                    'x2': round(x2, 2),
                    'y2': round(y2, 2),
                    'width': round(x2-x1, 2),
                    'height': round(y2-y1, 2)
                }
            })
        
        return json.dumps({
            'total_detections': len(detections),
            'detections': detections,
            'model': 'YOLOv8s (Phase 1 Improved)',
            'performance': 'mAP50=51.2%, Recall=55.7%',
            'timestamp': datetime.now().isoformat()
        }, indent=2, ensure_ascii=False)


def create_interface():
    """创建Gradio界面"""
    
    # 初始化检测器
    detector = DashboardDetector()
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', 'Microsoft YaHei', sans-serif;
    }
    """
    
    with gr.Blocks(
        title="仪表盘图表检测系统",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🎯 仪表盘图表检测系统
        
        **AI自动识别**仪表盘中的各种图表类型并精确定位
        
        📊 **模型性能**: mAP50 = 51.2%, Recall = 55.7%  
        🎯 **支持类型**: 20种图表（Bar chart, Line chart, Pie chart, Map, Card等）  
        ⚡ **技术栈**: YOLOv8s + GPU训练
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📤 上传仪表盘图像")
                
                input_image = gr.Image(
                    label="支持PNG/JPG格式",
                    type="numpy",
                    height=400
                )
                
                gr.Markdown("### ⚙️ 检测参数")
                
                conf_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.95,
                    value=0.10,
                    step=0.01,
                    label="置信度阈值",
                    info="越低检测越全面，建议0.05-0.25"
                )
                
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="NMS IoU阈值",
                    info="越低保留的框越少"
                )
                
                detect_btn = gr.Button(
                    "🚀 开始检测",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 使用提示
                
                **检测效果优秀**: Map, Bar chart, Bubble chart  
                **检测效果良好**: Card, Donut chart  
                **检测效果一般**: Line chart, Heatmap, Scatter plot
                
                **调优建议**:
                - Line chart/Heatmap: 降低置信度到**0.05**
                - Bar chart/Map: 默认0.10即可
                - 误检多: 提高置信度到0.20
                """)
            
            with gr.Column(scale=1):
                # 输出区域
                gr.Markdown("### 📊 检测结果")
                
                output_image = gr.Image(
                    label="标注后的图像（彩色边界框）",
                    type="numpy",
                    height=400
                )
                
                gr.Markdown("### 📝 检测摘要")
                
                summary_text = gr.Textbox(
                    label="",
                    lines=10,
                    show_label=False
                )
                
                with gr.Accordion("🔍 详细结果 (JSON格式)", open=False):
                    detail_json = gr.Code(
                        label="",
                        language="json",
                        lines=12
                    )
        
        # 支持的图表类型说明
        gr.Markdown("""
        ---
        ### 📋 支持的图表类型（20种）
        
        | 类别 | 检测效果 | 建议置信度 |
        |------|---------|-----------|
        | Map (地图) | ⭐⭐⭐⭐⭐ 95%+ | 0.10 |
        | Bar chart (柱状图) | ⭐⭐⭐⭐⭐ 85%+ | 0.10 |
        | Bubble chart (气泡图) | ⭐⭐⭐⭐⭐ 99%+ | 0.10 |
        | Card (卡片) | ⭐⭐⭐⭐ 60%+ | 0.10 |
        | Donut chart (环形图) | ⭐⭐⭐⭐ 58%+ | 0.10 |
        | Data table (数据表) | ⭐⭐⭐ 52%+ | 0.10 |
        | Area chart (面积图) | ⭐⭐⭐ 48%+ | 0.08 |
        | Scatter plot (散点图) | ⭐⭐⭐ 38%+ | 0.05 |
        | Line chart (折线图) | ⭐⭐ 32% | **0.05** |
        | Heatmap (热力图) | ⭐⭐ 22% | **0.05** |
        | Pie/Radar/Timeline等 | ⭐ 10-15% | **0.03** |
        
        ---
        ### 📌 关于本项目
        
        **训练数据**: 144张仪表盘图像，239个标注实例  
        **模型架构**: YOLOv8s (11M参数)  
        **训练设备**: NVIDIA GPU (CUDA 12.6)  
        **训练时长**: 111轮，约12分钟  
        **最佳性能**: mAP50 = 51.2%, Recall = 55.7%
        
        **改进方向**:
        - 收集更多Line chart和Heatmap样本
        - 升级到更大模型（YOLOv8m）
        - 引入注意力机制
        
        **开源代码**: [GitHub链接]  
        **反馈**: 欢迎提Issue或Pull Request
        """)
        
        # 绑定事件
        detect_btn.click(
            fn=detector.detect,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_text, detail_json]
        )
    
    return app


if __name__ == "__main__":
    print("=" * 70)
    print("🌐 仪表盘图表检测系统 - Hugging Face Spaces")
    print("=" * 70)
    print()
    
    # 创建并启动应用
    app = create_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Hugging Face会自动提供公开链接
    )

