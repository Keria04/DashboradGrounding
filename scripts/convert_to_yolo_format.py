#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将CVAT XML标注转换为YOLO格式
"""

import sys
import yaml
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xml_parser import CVATXMLParser

# 加载配置
config_path = Path(__file__).parent.parent / 'configs' / 'config_yolo.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

chart_types = config['model']['chart_types']
annotations_dir = config['data']['annotations_dir']
raw_dir = config['data']['raw_dir']

# 创建YOLO格式的目录结构
yolo_data_dir = Path('data/yolo_format')
yolo_images_dir = yolo_data_dir / 'images'
yolo_labels_dir = yolo_data_dir / 'labels'

for split in ['train', 'val', 'test']:
    (yolo_images_dir / split).mkdir(parents=True, exist_ok=True)
    (yolo_labels_dir / split).mkdir(parents=True, exist_ok=True)

print("="*70)
print("转换CVAT XML到YOLO格式")
print("="*70)

# 解析XML
parser = CVATXMLParser(chart_types)
annotations = parser.parse_directory(annotations_dir)

print(f"\n解析到 {len(annotations)} 个标注")

# 创建类别索引映射
class_to_idx = {chart_type: idx for idx, chart_type in enumerate(chart_types)}

# 数据分割（与训练时保持一致）
import numpy as np
np.random.seed(42)
indices = np.random.permutation(len(annotations))

train_size = int(len(annotations) * 0.7)
val_size = int(len(annotations) * 0.15)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

splits = {
    'train': train_indices,
    'val': val_indices,
    'test': test_indices
}

print(f"\n数据分割:")
print(f"  训练集: {len(train_indices)}")
print(f"  验证集: {len(val_indices)}")
print(f"  测试集: {len(test_indices)}")

# 转换每个标注
total_converted = 0

for split_name, split_indices in splits.items():
    print(f"\n处理 {split_name} 集...")
    
    for idx in split_indices:
        annotation = annotations[idx]
        
        # 复制图像
        src_image = Path(raw_dir) / annotation.image_name
        dst_image = yolo_images_dir / split_name / annotation.image_name
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        
        # 创建YOLO格式标注文件
        label_file = yolo_labels_dir / split_name / annotation.image_name.replace('.png', '.txt')
        
        with open(label_file, 'w', encoding='utf-8') as f:
            for bbox in annotation.bounding_boxes:
                # YOLO格式: <class_id> <x_center> <y_center> <width> <height>
                # 全部归一化到[0, 1]
                
                x_center = ((bbox.x1 + bbox.x2) / 2) / annotation.image_width
                y_center = ((bbox.y1 + bbox.y2) / 2) / annotation.image_height
                width = (bbox.x2 - bbox.x1) / annotation.image_width
                height = (bbox.y2 - bbox.y1) / annotation.image_height
                
                class_id = class_to_idx[bbox.label]
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        total_converted += 1

print(f"\n✓ 转换完成！共处理 {total_converted} 张图像")
print(f"\n数据保存在: {yolo_data_dir}")
print(f"  - images/train/, images/val/, images/test/")
print(f"  - labels/train/, labels/val/, labels/test/")

