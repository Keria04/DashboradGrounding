#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仪表盘图表检测与分割 - 交互式Demo应用
支持用户上传图片，自动检测并标注各种图表类型
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime

# 设置UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DashboardChartDetector:
    """仪表盘图表检测器"""
    
    # 颜色映射（20种不同颜色用于不同图表类型）
    COLORS = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 品红
        (0, 255, 255),    # 青色
        (255, 128, 0),    # 橙色
        (128, 0, 255),    # 紫色
        (0, 255, 128),    # 春绿
        (255, 0, 128),    # 玫瑰红
        (128, 255, 0),    # 黄绿
        (0, 128, 255),    # 天蓝
        (255, 128, 128),  # 浅红
        (128, 255, 128),  # 浅绿
        (128, 128, 255),  # 浅蓝
        (255, 255, 128),  # 浅黄
        (255, 128, 255),  # 浅品红
        (128, 255, 255),  # 浅青
        (192, 192, 192),  # 银色
        (255, 165, 0),    # 深橙
    ]
    
    def __init__(self, model_path=None, conf_threshold=0.10, iou_threshold=0.4):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，如果为None则自动查找最佳模型
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 自动查找模型
        if model_path is None:
            model_path = self._find_best_model()
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                f"未找到模型文件！\n"
                f"请先训练模型:\n"
                f"  python scripts/train_yolo_optimized.py\n"
                f"或手动指定模型路径:\n"
                f"  python scripts/app_demo.py --model path/to/best.pt"
            )
        
        print(f"📦 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = Path(model_path)
        
        # 获取类别名称
        self.class_names = self.model.names
        print(f"✅ 模型加载成功！支持 {len(self.class_names)} 种图表类型")
        print(f"   类别: {list(self.class_names.values())[:5]}... (共{len(self.class_names)}种)")
        print()
    
    def _find_best_model(self):
        """自动查找最佳模型"""
        # 优先级顺序
        search_paths = [
            PROJECT_ROOT / "experiments" / "yolov8s_optimized" / "weights" / "best.pt",
            PROJECT_ROOT / "experiments" / "ultralytics_yolo" / "weights" / "best.pt",
            PROJECT_ROOT / "experiments" / "yolo_phase1_*" / "weights" / "best.pt",
        ]
        
        for pattern in search_paths:
            if '*' in str(pattern):
                # 使用glob查找
                matches = list(pattern.parent.parent.parent.glob(pattern.name))
                if matches:
                    # 返回最新的
                    return max(matches, key=lambda p: p.stat().st_mtime)
            elif pattern.exists():
                return pattern
        
        return None
    
    def detect(self, image_path):
        """
        检测图像中的图表
        
        Args:
            image_path: 图像路径
            
        Returns:
            results: 检测结果
            annotated_image: 标注后的图像
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 执行检测
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # 标注图像
        annotated_image = self._annotate_image(image.copy(), results)
        
        return results, annotated_image
    
    def _annotate_image(self, image, results):
        """在图像上绘制检测框和标签"""
        boxes = results.boxes
        
        if len(boxes) == 0:
            # 如果没有检测到任何目标，添加提示
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
        
        # 遍历每个检测框
        for i, box in enumerate(boxes):
            # 获取坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # 获取类别和置信度
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = self.class_names[cls_id]
            
            # 选择颜色
            color = self.COLORS[cls_id % len(self.COLORS)]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = f"{class_name}: {conf:.2f}"
            
            # 计算标签背景大小
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            # 绘制标签背景
            cv2.rectangle(
                image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # 在框内添加序号
            cv2.putText(
                image,
                f"#{i+1}",
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
        
        return image
    
    def print_detection_summary(self, results):
        """打印检测摘要"""
        boxes = results.boxes
        
        if len(boxes) == 0:
            print("❌ 未检测到任何图表")
            print()
            print("💡 建议:")
            print("   - 降低置信度阈值 (--conf 0.05)")
            print("   - 检查图像质量")
            print("   - 确保图像包含支持的图表类型")
            return
        
        print(f"✅ 检测到 {len(boxes)} 个图表:")
        print()
        
        # 按类别分组统计
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[cls_id]
            conf = float(box.conf[0].cpu().numpy())
            
            if class_name not in class_counts:
                class_counts[class_name] = {'count': 0, 'confs': []}
            class_counts[class_name]['count'] += 1
            class_counts[class_name]['confs'].append(conf)
        
        # 打印统计
        for i, (class_name, info) in enumerate(sorted(
            class_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ), 1):
            avg_conf = np.mean(info['confs'])
            print(f"  {i}. {class_name}: {info['count']}个 (平均置信度: {avg_conf:.2%})")
        
        print()
    
    def save_results(self, image, output_path):
        """保存标注后的图像"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        print(f"💾 结果已保存: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="仪表盘图表检测与分割 - 交互式Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 检测单张图片
  python scripts/app_demo.py --image data/raw/dashboard_0001.png
  
  # 指定输出路径
  python scripts/app_demo.py --image data/raw/dashboard_0001.png --output results/result.png
  
  # 调整置信度阈值（提高召回率）
  python scripts/app_demo.py --image data/raw/dashboard_0001.png --conf 0.05
  
  # 使用指定模型
  python scripts/app_demo.py --image data/raw/dashboard_0001.png --model experiments/yolov8s_optimized/weights/best.pt
  
  # 批量处理（处理文件夹中所有图片）
  python scripts/app_demo.py --image data/raw --output results
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='输入图像路径（文件或文件夹）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图像路径（默认: output/detection_结果.png）'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='模型路径（默认: 自动查找最佳模型）'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.10,
        help='置信度阈值 (默认: 0.10)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.4,
        help='NMS的IoU阈值 (默认: 0.4)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示结果（需要GUI环境）'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("=" * 70)
    print("🎯 仪表盘图表检测与分割 - Phase 1 Demo")
    print("=" * 70)
    print()
    
    try:
        # 初始化检测器
        detector = DashboardChartDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # 处理输入
        input_path = Path(args.image)
        
        if not input_path.exists():
            print(f"❌ 错误: 输入路径不存在: {input_path}")
            return
        
        # 判断是文件还是文件夹
        if input_path.is_file():
            # 单文件处理
            image_files = [input_path]
        else:
            # 文件夹处理
            print(f"📁 扫描文件夹: {input_path}")
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_files.extend(input_path.glob(ext))
            print(f"   找到 {len(image_files)} 个图像文件")
            print()
        
        if len(image_files) == 0:
            print("❌ 未找到图像文件")
            return
        
        # 处理每个图像
        for i, image_file in enumerate(image_files, 1):
            print(f"{'=' * 70}")
            print(f"处理 [{i}/{len(image_files)}]: {image_file.name}")
            print(f"{'=' * 70}")
            print()
            
            # 检测
            results, annotated_image = detector.detect(image_file)
            
            # 打印摘要
            detector.print_detection_summary(results)
            
            # 确定输出路径
            if args.output:
                output_path = Path(args.output)
                if output_path.is_dir() or len(image_files) > 1:
                    # 如果是文件夹或批量处理
                    output_path = output_path / f"detected_{image_file.stem}.png"
            else:
                # 默认输出路径
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = PROJECT_ROOT / "output" / f"detection_{image_file.stem}_{timestamp}.png"
            
            # 保存结果
            detector.save_results(annotated_image, output_path)
            
            # 显示（如果需要）
            if args.show:
                cv2.imshow(f"Detection Result - {image_file.name}", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        print("=" * 70)
        print("✅ 所有图像处理完成！")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

