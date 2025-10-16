#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理现有数据到批次结构

将当前的data/raw和data/annotations目录下的文件
整理成batch_001的标准批次结构

用法:
    python scripts/organize_existing_data.py
    python scripts/organize_existing_data.py --dry-run  # 仅预览，不实际移动
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def organize_data(dry_run=False):
    """整理现有数据"""
    
    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw"
    ann_dir = data_dir / "annotations"
    
    # 新的批次目录
    batch_001_raw = raw_dir / "batch_001"
    batch_001_ann = ann_dir / "batch_001"
    
    print(f"{'🔍 [DRY RUN] ' if dry_run else ''}开始整理现有数据到batch_001结构")
    print("=" * 70)
    
    # 1. 整理raw目录
    print(f"\n📁 整理原始图片...")
    
    if not batch_001_raw.exists():
        print(f"   创建目录: {batch_001_raw}")
        if not dry_run:
            batch_001_raw.mkdir(parents=True, exist_ok=True)
    
    # 移动PNG文件（排除已在batch_*目录中的）
    png_files = []
    for f in raw_dir.glob("*.png"):
        if f.is_file():
            png_files.append(f)
    
    if png_files:
        print(f"   找到 {len(png_files)} 个PNG文件需要移动")
        if not dry_run:
            for f in png_files:
                dest = batch_001_raw / f.name
                shutil.move(str(f), str(dest))
                print(f"   移动: {f.name}")
        else:
            print(f"   [DRY RUN] 将移动 {len(png_files)} 个文件到 {batch_001_raw}")
    else:
        print(f"   ✅ 没有需要移动的PNG文件（可能已经整理过）")
    
    # 2. 整理annotations目录
    print(f"\n📋 整理标注文件...")
    
    if not batch_001_ann.exists():
        print(f"   创建目录: {batch_001_ann}")
        if not dry_run:
            batch_001_ann.mkdir(parents=True, exist_ok=True)
    
    # 移动annotator目录（排除已在batch_*中的和new annotations）
    annotator_dirs = []
    for d in ann_dir.iterdir():
        if d.is_dir() and d.name.startswith("annotator") and not d.parent.name.startswith("batch_"):
            annotator_dirs.append(d)
    
    if annotator_dirs:
        print(f"   找到 {len(annotator_dirs)} 个标注者目录需要移动")
        if not dry_run:
            for d in annotator_dirs:
                dest = batch_001_ann / d.name
                if dest.exists():
                    print(f"   ⚠️  目标已存在，跳过: {d.name}")
                else:
                    shutil.move(str(d), str(dest))
                    print(f"   移动: {d.name}")
        else:
            print(f"   [DRY RUN] 将移动以下目录到 {batch_001_ann}:")
            for d in annotator_dirs:
                print(f"      - {d.name}")
    else:
        print(f"   ✅ 没有需要移动的标注目录（可能已经整理过）")
    
    # 3. 显示最终结构
    print(f"\n📊 整理{'预览' if dry_run else '完成'}:")
    print(f"\n   {batch_001_raw}/")
    if batch_001_raw.exists():
        png_count = len(list(batch_001_raw.glob("*.png")))
        print(f"      ├─ {png_count} 个PNG文件")
    
    print(f"\n   {batch_001_ann}/")
    if batch_001_ann.exists():
        for d in sorted(batch_001_ann.iterdir()):
            if d.is_dir():
                xml_count = len(list(d.glob("*.xml")))
                print(f"      ├─ {d.name}/ ({xml_count} 个XML文件)")
    
    print("\n" + "=" * 70)
    
    if not dry_run:
        print(f"\n✅ 整理完成！")
        print(f"\n📝 后续步骤:")
        print(f"   1. 验证数据: python scripts/validate_data.py --batch batch_001")
        print(f"   2. 查看统计: python scripts/show_data_stats.py --detailed")
    else:
        print(f"\n🔍 这是预览模式，未实际移动文件")
        print(f"   运行 python scripts/organize_existing_data.py 来执行实际操作")


def main():
    parser = argparse.ArgumentParser(
        description="整理现有数据到batch_001批次结构"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅预览，不实际移动文件'
    )
    
    args = parser.parse_args()
    
    organize_data(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

