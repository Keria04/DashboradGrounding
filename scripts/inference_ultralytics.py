#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用官方ultralytics YOLO进行推理
实现第一阶段的两个任务
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.encoding_config import initialize_encoding
    initialize_encoding()
except:
    pass

from ultralytics import YOLO
import cv2
from PIL import Image

class UltralyticsInference:
    """官方YOLO推理类"""
    
    def __init__(self, model_path: str):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径（.pt文件）
        """
        self.model = YOLO(model_path)
        
        # 图表类型名称
        self.chart_types = [
            "Area chart", "Bar chart", "Bubble chart", "Calendar", "Card",
            "Data table", "Dial chart", "Donut chart", "Gantt chart", "Heatmap",
            "Line chart", "Map", "Mind Map", "Percentage indicator", "Pie chart",
            "Progress bar", "Radar chart", "Scatter plot", "Time Line", "Waterfall chart"
        ]
        
        print(f"✓ 模型加载成功")
        print(f"  支持 {len(self.chart_types)} 种图表类型")
    
    def detect_charts(self, image_path: str, conf=0.25):
        """
        任务1: 检测图表并分类
        
        Args:
            image_path: 图像路径
            conf: 置信度阈值
            
        Returns:
            结果字典
        """
        # 推理
        results = self.model.predict(image_path, conf=conf, verbose=False)
        
        # 解析结果
        result = results[0]
        boxes = result.boxes
        
        # 提取信息
        xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        # 映射到图表类型名称
        labels = [self.chart_types[cid] for cid in class_ids]
        
        return {
            'boxes': xyxy,
            'labels': labels,
            'scores': confidences,
            'image_path': image_path,
            'class_ids': class_ids
        }
    
    def find_chart_by_text(self, image_path: str, query_text: str, conf=0.25):
        """
        任务2: 文本查询定位图表
        
        Args:
            image_path: 图像路径
            query_text: 查询文本
            conf: 置信度阈值
            
        Returns:
            匹配结果
        """
        # 先检测
        detection = self.detect_charts(image_path, conf)
        
        # 关键词映射
        keyword_mappings = {
            'line': ['折线', 'line', '线图'],
            'bar': ['柱状', 'bar', '柱图', '条形'],
            'pie': ['饼图', 'pie', '饼'],
            'area': ['面积', 'area', '区域'],
            'bubble': ['气泡', 'bubble'],
            'calendar': ['日历', 'calendar'],
            'card': ['卡片', 'card'],
            'table': ['表格', 'table', '数据表'],
            'dial': ['仪表', 'dial', '表盘'],
            'donut': ['环形', 'donut', '甜甜圈'],
            'gantt': ['甘特', 'gantt'],
            'heatmap': ['热力', 'heatmap', '热图'],
            'map': ['地图', 'map'],
            'mind': ['思维', 'mind', '脑图'],
            'percentage': ['百分比', 'percentage', '指示器'],
            'progress': ['进度', 'progress'],
            'radar': ['雷达', 'radar'],
            'scatter': ['散点', 'scatter'],
            'timeline': ['时间', 'timeline', '时间线'],
            'waterfall': ['瀑布', 'waterfall']
        }
        
        query_lower = query_text.lower()
        
        # 匹配
        matched_charts = []
        for i, (box, label, score) in enumerate(zip(
            detection['boxes'],
            detection['labels'],
            detection['scores']
        )):
            label_lower = label.lower()
            
            # 检查匹配
            for chart_key, keywords in keyword_mappings.items():
                if any(kw in query_lower for kw in keywords):
                    if chart_key in label_lower:
                        matched_charts.append({
                            'box': box.tolist(),
                            'label': label,
                            'score': float(score),
                            'index': i
                        })
                        break
        
        return {
            'matched_charts': matched_charts,
            'query': query_text,
            'total_detections': len(detection['labels']),
            'all_detections': detection
        }
    
    def visualize(self, image_path: str, save_path: str = None, conf=0.25):
        """可视化检测结果"""
        results = self.model.predict(image_path, conf=conf, save=False)
        
        # 绘制结果
        annotated = results[0].plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"✓ 可视化保存到: {save_path}")
        else:
            from matplotlib import pyplot as plt
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


def main():
    """演示用法"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--image', required=True, help='图像路径')
    parser.add_argument('--query', default=None, help='文本查询')
    parser.add_argument('--output', default=None, help='输出路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = UltralyticsInference(args.model)
    
    if args.query:
        # 任务2: 文本查询
        result = inference.find_chart_by_text(args.image, args.query, args.conf)
        print(f"\n查询: '{result['query']}'")
        print(f"找到 {len(result['matched_charts'])} 个匹配")
        for chart in result['matched_charts']:
            print(f"  - {chart['label']} ({chart['score']:.2%})")
    else:
        # 任务1: 检测
        result = inference.detect_charts(args.image, args.conf)
        print(f"\n检测到 {len(result['boxes'])} 个图表")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  - {label}: {score:.2%}")
    
    # 可视化
    if args.output:
        inference.visualize(args.image, args.output, args.conf)

if __name__ == "__main__":
    main()

