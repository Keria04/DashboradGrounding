#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仪表盘图表检测 - Web界面
提供友好的浏览器界面，支持拖拽上传、实时预览、批量处理
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
from datetime import datetime
import json

# 设置UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class WebDashboardDetector:
    """Web版本的仪表盘检测器"""
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
        (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128),
        (255, 128, 255), (128, 255, 255), (192, 192, 192), (255, 165, 0),
    ]
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = None
    
    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = self._find_best_model()
        
        if model_path is None or not Path(model_path).exists():
            return False, "未找到模型文件！请先训练模型。"
        
        try:
            self.model = YOLO(str(model_path))
            self.model_path = Path(model_path)
            self.class_names = self.model.names
            return True, f"✅ 模型加载成功！支持 {len(self.class_names)} 种图表类型"
        except Exception as e:
            return False, f"❌ 模型加载失败: {e}"
    
    def _find_best_model(self):
        """查找最佳模型"""
        search_paths = [
            PROJECT_ROOT / "experiments" / "yolov8s_optimized" / "weights" / "best.pt",
            PROJECT_ROOT / "experiments" / "ultralytics_yolo" / "weights" / "best.pt",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # 搜索所有experiment文件夹
        exp_dir = PROJECT_ROOT / "experiments"
        if exp_dir.exists():
            best_pts = list(exp_dir.glob("*/weights/best.pt"))
            if best_pts:
                return max(best_pts, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def detect(self, image, conf_threshold=0.10, iou_threshold=0.4):
        """
        检测图表
        
        Returns:
            annotated_image: 标注后的图像
            summary_text: 检测摘要文本
            detail_json: 详细结果JSON
        """
        if self.model is None:
            success, msg = self.load_model()
            if not success:
                return None, msg, "{}"
        
        if image is None:
            return None, "请先上传图像", "{}"
        
        try:
            # 执行检测
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # 标注图像
            annotated_image = self._annotate_image(image.copy(), results)
            
            # 生成摘要
            summary_text = self._generate_summary(results)
            
            # 生成详细JSON
            detail_json = self._generate_detail_json(results)
            
            return annotated_image, summary_text, detail_json
            
        except Exception as e:
            return None, f"❌ 检测失败: {e}", "{}"
    
    def _annotate_image(self, image, results):
        """标注图像"""
        boxes = results.boxes
        
        if len(boxes) == 0:
            h, w = image.shape[:2]
            cv2.putText(
                image,
                "No charts detected!",
                (w // 2 - 150, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            return image
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = self.class_names[cls_id]
            
            color = self.COLORS[cls_id % len(self.COLORS)]
            
            # 绘制框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # 标签
            label = f"#{i+1} {class_name}: {conf:.2%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(
                image,
                (x1, y1 - label_h - baseline - 8),
                (x1 + label_w + 10, y1),
                color,
                -1
            )
            
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return image
    
    def _generate_summary(self, results):
        """生成摘要文本"""
        boxes = results.boxes
        
        if len(boxes) == 0:
            return "❌ 未检测到任何图表\n\n💡 建议:\n- 降低置信度阈值\n- 检查图像质量\n- 确保包含支持的图表类型"
        
        summary = f"✅ 检测到 {len(boxes)} 个图表:\n\n"
        
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
        
        # 生成统计
        for i, (class_name, info) in enumerate(sorted(
            class_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ), 1):
            avg_conf = np.mean(info['confs'])
            summary += f"  {i}. **{class_name}**: {info['count']}个 (平均置信度: {avg_conf:.1%})\n"
        
        return summary
    
    def _generate_detail_json(self, results):
        """生成详细JSON"""
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
                    'width': round(x2 - x1, 2),
                    'height': round(y2 - y1, 2)
                }
            })
        
        return json.dumps({
            'total_detections': len(detections),
            'detections': detections,
            'model': str(self.model_path.name) if self.model_path else 'Unknown',
            'timestamp': datetime.now().isoformat()
        }, indent=2, ensure_ascii=False)


def create_web_app():
    """创建Gradio Web应用"""
    
    detector = WebDashboardDetector()
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-image {
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    """
    
    with gr.Blocks(
        title="仪表盘图表检测系统",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🎯 仪表盘图表检测与分割系统
        
        **Phase 1**: 自动识别仪表盘中的各种图表类型并进行精确分割
        
        支持的图表类型: Bar chart, Line chart, Pie chart, Area chart, Scatter plot, Heatmap, 
        Map, Card, Data table, Donut chart, Bubble chart等20种类型
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📤 上传图像")
                input_image = gr.Image(
                    label="拖拽或点击上传仪表盘图像",
                    type="numpy",
                    height=400
                )
                
                gr.Markdown("### ⚙️ 检测参数")
                conf_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.95,
                    value=0.10,
                    step=0.01,
                    label="置信度阈值（越低召回率越高，但误检也会增加）",
                    info="建议范围: 0.05-0.25"
                )
                
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="NMS IoU阈值（越低保留的框越少）",
                    info="建议范围: 0.3-0.5"
                )
                
                detect_btn = gr.Button(
                    "🚀 开始检测",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 使用提示
                - 支持 PNG, JPG, JPEG 格式
                - 建议图像分辨率: 640x640 以上
                - 置信度阈值越低，检测越全面（但可能有误检）
                - 单击"开始检测"执行分析
                """)
            
            with gr.Column(scale=1):
                # 输出区域
                gr.Markdown("### 📊 检测结果")
                output_image = gr.Image(
                    label="标注后的图像",
                    type="numpy",
                    height=400,
                    elem_classes=["output-image"]
                )
                
                gr.Markdown("### 📝 检测摘要")
                summary_text = gr.Textbox(
                    label="",
                    lines=8,
                    max_lines=15,
                    show_label=False
                )
                
                with gr.Accordion("🔍 详细结果 (JSON)", open=False):
                    detail_json = gr.Code(
                        label="",
                        language="json",
                        lines=10
                    )
        
        # 示例
        gr.Markdown("### 📸 示例图像")
        
        # 查找示例图像
        example_images = []
        raw_dir = PROJECT_ROOT / "data" / "raw"
        if raw_dir.exists():
            example_files = list(raw_dir.glob("dashboard_*.png"))[:6]
            example_images = [[str(f)] for f in example_files]
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=input_image,
                label="点击加载示例"
            )
        
        # 绑定事件
        detect_btn.click(
            fn=detector.detect,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, summary_text, detail_json]
        )
        
        gr.Markdown("""
        ---
        ### 📌 关于系统
        
        **模型**: Ultralytics YOLOv8 (优化版)  
        **训练数据**: 144 张仪表盘图像，239 个标注实例  
        **性能**: mAP50 ≥ 60% (目标)
        
        **下一步改进**:
        - 增加更多训练数据（尤其是Line chart, Heatmap等困难类别）
        - 升级到更大模型（YOLOv8m/l）
        - 集成Phase 2功能（文本查询定位）
        
        📧 如有问题或建议，请联系开发团队
        """)
    
    return app


def main():
    """启动Web应用"""
    print("=" * 70)
    print("🚀 启动仪表盘图表检测Web应用...")
    print("=" * 70)
    print()
    
    # 创建应用
    app = create_web_app()
    
    # 启动服务器
    print("✅ Web应用已准备就绪！")
    print()
    print("🌐 访问地址:")
    print("   本地: http://127.0.0.1:7860")
    print("   网络: http://0.0.0.0:7860")
    print()
    print("💡 提示: 按 Ctrl+C 停止服务器")
    print("=" * 70)
    print()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # ✅ 生成公网链接（72小时有效）
        inbrowser=True  # 自动打开浏览器
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n💡 请确保已安装 gradio:")
        print("   pip install gradio")

