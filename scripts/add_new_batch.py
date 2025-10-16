#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新批次数据自动添加工具

功能:
1. 验证新批次数据的规范性
2. 自动转换为YOLO格式
3. 更新数据索引
4. 生成统计报告

用法:
    python scripts/add_new_batch.py --batch batch_002
    python scripts/add_new_batch.py --batch batch_002 --dry-run  # 仅检查不转换
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil
from collections import defaultdict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BatchManager:
    """批次数据管理器"""
    
    def __init__(self, batch_id, dry_run=False):
        self.batch_id = batch_id
        self.dry_run = dry_run
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        
        # 路径
        self.raw_batch_dir = self.data_dir / "raw" / batch_id
        self.ann_batch_dir = self.data_dir / "annotations" / batch_id
        self.stats_dir = self.data_dir / "statistics"
        
        # 确保统计目录存在
        self.stats_dir.mkdir(exist_ok=True, parents=True)
        
        self.issues = []
        self.warnings = []
        
    def validate_batch_structure(self):
        """验证批次目录结构"""
        print(f"\n📁 验证批次 {self.batch_id} 的目录结构...")
        
        # 检查raw目录
        if not self.raw_batch_dir.exists():
            self.issues.append(f"❌ 原始图片目录不存在: {self.raw_batch_dir}")
            return False
        
        # 检查annotations目录
        if not self.ann_batch_dir.exists():
            self.issues.append(f"❌ 标注目录不存在: {self.ann_batch_dir}")
            return False
        
        print(f"✅ 目录结构正确")
        return True
    
    def collect_images(self):
        """收集图片文件"""
        print(f"\n🖼️  收集图片文件...")
        
        images = sorted(self.raw_batch_dir.glob("*.png"))
        
        if not images:
            self.issues.append(f"❌ 未找到PNG图片: {self.raw_batch_dir}")
            return []
        
        print(f"✅ 找到 {len(images)} 张图片")
        
        # 验证命名规范
        naming_errors = []
        for img in images:
            if not img.stem.startswith("dashboard_"):
                naming_errors.append(img.name)
        
        if naming_errors:
            self.warnings.append(f"⚠️  以下图片命名不符合规范（应为dashboard_XXXX.png）:")
            for name in naming_errors[:10]:  # 只显示前10个
                self.warnings.append(f"    - {name}")
        
        return images
    
    def collect_annotations(self):
        """收集标注文件"""
        print(f"\n📋 收集标注文件...")
        
        annotations = []
        for annotator_dir in self.ann_batch_dir.iterdir():
            if annotator_dir.is_dir():
                xml_files = list(annotator_dir.glob("*.xml"))
                annotations.extend(xml_files)
        
        if not annotations:
            self.issues.append(f"❌ 未找到XML标注文件: {self.ann_batch_dir}")
            return []
        
        print(f"✅ 找到 {len(annotations)} 个标注文件")
        return annotations
    
    def check_image_annotation_match(self, images, annotations):
        """检查图片和标注的对应关系"""
        print(f"\n🔗 检查图片-标注对应关系...")
        
        # 简化检查：确保有图片和标注即可
        # 实际对应关系在convert_to_yolo_format.py中处理
        
        if images and annotations:
            print(f"✅ 图片和标注文件都存在")
            return True
        else:
            self.issues.append("❌ 图片或标注文件缺失")
            return False
    
    def get_batch_info(self, images):
        """获取批次信息"""
        print(f"\n📊 收集批次统计信息...")
        
        info = {
            "batch_id": self.batch_id,
            "num_images": len(images),
            "date_added": datetime.now().strftime("%Y-%m-%d"),
            "images": []
        }
        
        # 提取图片编号范围
        img_ids = []
        for img in images:
            try:
                img_id = int(img.stem.split('_')[1])
                img_ids.append(img_id)
            except:
                pass
        
        if img_ids:
            info["id_range"] = {
                "min": min(img_ids),
                "max": max(img_ids)
            }
        
        print(f"✅ 批次信息收集完成")
        print(f"   - 图片数量: {info['num_images']}")
        if "id_range" in info:
            print(f"   - ID范围: {info['id_range']['min']} ~ {info['id_range']['max']}")
        
        return info
    
    def update_batch_index(self, batch_info):
        """更新批次索引"""
        index_file = self.stats_dir / "batch_info.json"
        
        # 读取现有索引
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"batches": {}}
        
        # 添加新批次
        data["batches"][self.batch_id] = batch_info
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["total_batches"] = len(data["batches"])
        
        # 保存
        if not self.dry_run:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 批次索引已更新: {index_file}")
        else:
            print(f"\n🔍 [DRY RUN] 将更新批次索引: {index_file}")
    
    def convert_to_yolo(self):
        """转换为YOLO格式"""
        print(f"\n🔄 转换为YOLO格式...")
        
        if self.dry_run:
            print(f"🔍 [DRY RUN] 跳过实际转换")
            return True
        
        # 调用原有的转换脚本
        convert_script = self.project_root / "scripts" / "convert_to_yolo_format.py"
        
        if convert_script.exists():
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(convert_script)],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                
                if result.returncode == 0:
                    print(f"✅ YOLO格式转换成功")
                    return True
                else:
                    print(f"❌ YOLO转换失败:")
                    print(result.stderr)
                    return False
                    
            except Exception as e:
                print(f"❌ 转换脚本执行错误: {e}")
                return False
        else:
            self.warnings.append(f"⚠️  转换脚本不存在: {convert_script}")
            return False
    
    def print_report(self):
        """打印验证报告"""
        print("\n" + "=" * 70)
        print(f"📊 批次 {self.batch_id} 验证报告")
        print("=" * 70)
        
        if self.issues:
            print("\n❌ 发现问题:")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print("\n⚠️  警告:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ 所有检查通过！")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """执行完整的添加流程"""
        print(f"\n{'🔍 [DRY RUN模式] ' if self.dry_run else ''}开始处理批次: {self.batch_id}")
        print("=" * 70)
        
        # 1. 验证目录结构
        if not self.validate_batch_structure():
            self.print_report()
            return False
        
        # 2. 收集图片
        images = self.collect_images()
        if not images:
            self.print_report()
            return False
        
        # 3. 收集标注
        annotations = self.collect_annotations()
        if not annotations:
            self.print_report()
            return False
        
        # 4. 检查对应关系
        if not self.check_image_annotation_match(images, annotations):
            self.print_report()
            return False
        
        # 5. 获取批次信息
        batch_info = self.get_batch_info(images)
        
        # 6. 更新索引
        self.update_batch_index(batch_info)
        
        # 7. 转换为YOLO格式
        if not self.convert_to_yolo():
            self.warnings.append("⚠️  YOLO转换可能需要手动运行")
        
        # 8. 打印报告
        self.print_report()
        
        if not self.issues:
            print(f"\n✅ 批次 {self.batch_id} 添加{'模拟' if self.dry_run else ''}完成！")
            print(f"\n📝 后续步骤:")
            print(f"   1. 查看数据统计: python scripts/show_data_stats.py")
            print(f"   2. 验证数据: python scripts/validate_data.py --batch {self.batch_id}")
            print(f"   3. 重新训练: START_PHASE1_IMPROVED_TRAINING.bat")
            return True
        else:
            print(f"\n❌ 批次 {self.batch_id} 验证失败，请修复上述问题")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="添加新批次数据到训练集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加新批次
  python scripts/add_new_batch.py --batch batch_002
  
  # 仅检查不执行
  python scripts/add_new_batch.py --batch batch_002 --dry-run
        """
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        required=True,
        help='批次ID (例如: batch_002)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅检查验证，不实际转换数据'
    )
    
    args = parser.parse_args()
    
    # 创建管理器并执行
    manager = BatchManager(args.batch, dry_run=args.dry_run)
    success = manager.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

