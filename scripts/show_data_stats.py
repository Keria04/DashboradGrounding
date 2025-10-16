#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计信息展示工具

功能:
1. 显示总体数据量
2. 各类别分布统计
3. 各批次信息
4. 数据集划分比例
5. 生成可视化图表

用法:
    python scripts/show_data_stats.py
    python scripts/show_data_stats.py --detailed
    python scripts/show_data_stats.py --export report.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DataStatistics:
    """数据统计分析器"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.stats_dir = self.data_dir / "statistics"
        self.yolo_dir = self.data_dir / "yolo_format"
        
        self.stats = {
            "overview": {},
            "classes": {},
            "batches": {},
            "splits": {}
        }
    
    def count_raw_images(self):
        """统计原始图片数量"""
        raw_dir = self.data_dir / "raw"
        
        if not raw_dir.exists():
            return 0, {}
        
        total = 0
        batch_counts = {}
        
        # 遍历批次目录
        for batch_dir in raw_dir.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith("batch_"):
                count = len(list(batch_dir.glob("*.png")))
                batch_counts[batch_dir.name] = count
                total += count
        
        # 如果没有批次目录，直接统计raw目录
        if not batch_counts:
            total = len(list(raw_dir.glob("*.png")))
            batch_counts["default"] = total
        
        return total, batch_counts
    
    def count_yolo_labels(self):
        """统计YOLO标签文件并分析类别分布"""
        labels_dir = self.yolo_dir / "labels"
        
        if not labels_dir.exists():
            return {}, {}
        
        class_counts = defaultdict(int)
        split_counts = {"train": 0, "val": 0, "test": 0}
        
        # 读取类别名称
        config_file = self.yolo_dir / "dashboard.yaml"
        class_names = []
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                class_names = config.get('names', [])
        
        # 统计各split的标签
        for split in ["train", "val", "test"]:
            split_dir = labels_dir / split
            if split_dir.exists():
                label_files = list(split_dir.glob("*.txt"))
                split_counts[split] = len(label_files)
                
                # 统计类别
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id < len(class_names):
                                    class_counts[class_names[class_id]] += 1
                                else:
                                    class_counts[f"Unknown_{class_id}"] += 1
        
        return dict(class_counts), split_counts
    
    def load_batch_info(self):
        """加载批次信息"""
        batch_file = self.stats_dir / "batch_info.json"
        
        if batch_file.exists():
            with open(batch_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("batches", {})
        
        return {}
    
    def collect_stats(self):
        """收集所有统计信息"""
        print("📊 正在收集数据统计信息...")
        
        # 原始图片统计
        total_images, batch_counts = self.count_raw_images()
        self.stats["overview"]["total_raw_images"] = total_images
        self.stats["batches"] = {
            "counts": batch_counts,
            "total_batches": len(batch_counts)
        }
        
        # YOLO数据统计
        class_dist, split_dist = self.count_yolo_labels()
        self.stats["classes"] = class_dist
        self.stats["splits"] = split_dist
        self.stats["overview"]["total_yolo_labels"] = sum(split_dist.values())
        self.stats["overview"]["total_objects"] = sum(class_dist.values())
        
        # 批次详细信息
        batch_info = self.load_batch_info()
        self.stats["batches"]["details"] = batch_info
        
        print("✅ 统计完成\n")
    
    def print_overview(self):
        """打印总览"""
        print("=" * 70)
        print("📊 数据集总览")
        print("=" * 70)
        
        overview = self.stats["overview"]
        print(f"\n📦 原始数据:")
        print(f"   总图片数: {overview.get('total_raw_images', 0)} 张")
        print(f"   总批次数: {self.stats['batches'].get('total_batches', 0)} 个")
        
        print(f"\n🎯 训练数据:")
        print(f"   YOLO标签文件: {overview.get('total_yolo_labels', 0)} 个")
        print(f"   标注对象总数: {overview.get('total_objects', 0)} 个")
        
        splits = self.stats["splits"]
        if sum(splits.values()) > 0:
            print(f"\n📊 数据集划分:")
            total = sum(splits.values())
            for split, count in splits.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   {split:6s}: {count:4d} ({percentage:5.1f}%)")
    
    def print_batches(self):
        """打印批次信息"""
        print("\n" + "=" * 70)
        print("📁 批次统计")
        print("=" * 70)
        
        batch_counts = self.stats["batches"].get("counts", {})
        batch_details = self.stats["batches"].get("details", {})
        
        if not batch_counts:
            print("\n⚠️  未找到批次数据")
            return
        
        print(f"\n批次数量: {len(batch_counts)}")
        print(f"\n{'批次ID':<15} {'图片数':<10} {'添加日期':<12} {'ID范围'}")
        print("-" * 70)
        
        for batch_id in sorted(batch_counts.keys()):
            count = batch_counts[batch_id]
            details = batch_details.get(batch_id, {})
            date = details.get("date_added", "N/A")
            id_range = details.get("id_range", {})
            
            if id_range:
                range_str = f"{id_range.get('min', '?')} ~ {id_range.get('max', '?')}"
            else:
                range_str = "N/A"
            
            print(f"{batch_id:<15} {count:<10} {date:<12} {range_str}")
    
    def print_class_distribution(self, detailed=False):
        """打印类别分布"""
        print("\n" + "=" * 70)
        print("📊 类别分布统计")
        print("=" * 70)
        
        class_dist = self.stats["classes"]
        
        if not class_dist:
            print("\n⚠️  未找到类别统计数据")
            return
        
        # 排序（按数量降序）
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
        
        total = sum(class_dist.values())
        print(f"\n总标注对象: {total} 个")
        print(f"类别数量: {len(class_dist)} 种")
        
        print(f"\n{'类别':<25} {'数量':<8} {'占比':<8} {'等级'}")
        print("-" * 70)
        
        for class_name, count in sorted_classes:
            percentage = (count / total * 100) if total > 0 else 0
            
            # 评级
            if count >= 50:
                level = "✅ 充足"
            elif count >= 20:
                level = "🟡 适中"
            elif count >= 10:
                level = "🟠 偏少"
            else:
                level = "🔴 紧缺"
            
            print(f"{class_name:<25} {count:<8} {percentage:5.1f}%  {level}")
        
        if detailed:
            self.print_priority_analysis(sorted_classes)
    
    def print_priority_analysis(self, sorted_classes):
        """打印优先级分析"""
        print("\n" + "=" * 70)
        print("🎯 数据收集优先级分析")
        print("=" * 70)
        
        # 分类
        urgent = []      # <10
        important = []   # 10-20
        moderate = []    # 20-50
        sufficient = []  # >=50
        
        for class_name, count in sorted_classes:
            if count < 10:
                urgent.append((class_name, count))
            elif count < 20:
                important.append((class_name, count))
            elif count < 50:
                moderate.append((class_name, count))
            else:
                sufficient.append((class_name, count))
        
        if urgent:
            print(f"\n🔴 紧急需要 (样本<10, 共{len(urgent)}类):")
            for name, count in urgent:
                target = max(30, count * 3)
                print(f"   - {name:<20} 当前:{count:3d}个  →  目标:{target:3d}+ (需+{target-count})")
        
        if important:
            print(f"\n🟡 重要补充 (样本10-20, 共{len(important)}类):")
            for name, count in important:
                target = 40
                print(f"   - {name:<20} 当前:{count:3d}个  →  目标:{target:3d}+ (需+{target-count})")
        
        if moderate:
            print(f"\n🟢 适量增加 (样本20-50, 共{len(moderate)}类):")
            for name, count in moderate:
                print(f"   - {name:<20} 当前:{count:3d}个  →  维持或小幅增加")
        
        if sufficient:
            print(f"\n✅ 样本充足 (样本≥50, 共{len(sufficient)}类):")
            print(f"   可维持当前水平，无需优先收集")
    
    def export_report(self, filepath):
        """导出统计报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # 重定向print输出
            import contextlib
            
            @contextlib.contextmanager
            def redirect_stdout(fileobj):
                old_stdout = sys.stdout
                sys.stdout = fileobj
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
            
            with redirect_stdout(f):
                f.write(f"数据集统计报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 70 + "\n\n")
                
                self.print_overview()
                self.print_batches()
                self.print_class_distribution(detailed=True)
        
        print(f"\n✅ 报告已导出: {filepath}")
    
    def run(self, detailed=False, export=None):
        """运行统计分析"""
        self.collect_stats()
        self.print_overview()
        self.print_batches()
        self.print_class_distribution(detailed=detailed)
        
        if export:
            self.export_report(export)
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="显示数据集统计信息",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='显示详细的优先级分析'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        metavar='FILE',
        help='导出报告到文件（如: report.txt）'
    )
    
    args = parser.parse_args()
    
    stats = DataStatistics()
    stats.run(detailed=args.detailed, export=args.export)


if __name__ == "__main__":
    main()

