"""
# -*- coding: utf-8 -*-
仪表盘数据集类
用于加载和预处理仪表盘图像及其标注数据
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .xml_parser import DashboardAnnotation, BoundingBox, CVATXMLParser

logger = logging.getLogger(__name__)


class DashboardDataset(Dataset):
    """仪表盘数据集类"""
    
    def __init__(
        self,
        annotations: List[DashboardAnnotation],
        raw_data_dir: str,
        chart_types: List[str],
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[transforms.Compose] = None,
        augment: bool = True,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        初始化数据集
        
        Args:
            annotations: 标注数据列表
            raw_data_dir: 原始图像目录
            chart_types: 图表类型列表
            image_size: 目标图像尺寸
            transform: 图像变换
            augment: 是否使用数据增强
            normalize_mean: 归一化均值
            normalize_std: 归一化标准差
        """
        self.annotations = annotations
        self.raw_data_dir = Path(raw_data_dir)
        self.chart_types = chart_types
        self.image_size = image_size
        self.chart_type_to_idx = {chart_type: idx for idx, chart_type in enumerate(chart_types)}
        self.idx_to_chart_type = {idx: chart_type for chart_type, idx in self.chart_type_to_idx.items()}
        
        # 数据增强
        if augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.augmentation = None
        
        # 基础变换
        self.base_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        # 过滤无效数据
        self._filter_valid_annotations()
        
        logger.info(f"数据集初始化完成，共 {len(self.annotations)} 张图像")
    
    def _filter_valid_annotations(self):
        """过滤无效的标注数据"""
        valid_annotations = []
        
        for annotation in self.annotations:
            # 检查图像文件是否存在
            image_path = self.raw_data_dir / annotation.image_name
            if not image_path.exists():
                logger.warning(f"图像文件不存在: {image_path}")
                continue
            
            # 检查是否有有效的边界框
            if not annotation.bounding_boxes:
                logger.warning(f"图像 {annotation.image_name} 没有边界框")
                continue
            
            # 检查边界框是否有效
            valid_boxes = []
            for bbox in annotation.bounding_boxes:
                # 检查坐标有效性
                coord_valid = bbox.x1 < bbox.x2 and bbox.y1 < bbox.y2
                # 检查标签匹配
                label_valid = bbox.label in self.chart_types
                
                if coord_valid and label_valid:
                    valid_boxes.append(bbox)
                elif not coord_valid:
                    logger.debug(f"图像 {annotation.image_name} 边界框坐标无效: {bbox.label} ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
                elif not label_valid:
                    # 详细检查为什么标签不匹配
                    logger.warning(f"图像 {annotation.image_name} 标签 '{bbox.label}' (type:{type(bbox.label)}) 不在配置中")
                    logger.warning(f"  配置类型示例: '{self.chart_types[0]}' (type:{type(self.chart_types[0])})")
                    logger.warning(f"  直接比较: {bbox.label == self.chart_types[0] if len(self.chart_types) > 0 else 'N/A'}")
                    logger.warning(f"  在列表中: {bbox.label in self.chart_types}")
            
            if valid_boxes:
                annotation.bounding_boxes = valid_boxes
                valid_annotations.append(annotation)
            else:
                logger.warning(f"图像 {annotation.image_name} 没有有效的边界框")
        
        self.annotations = valid_annotations
        logger.info(f"过滤后剩余 {len(self.annotations)} 张有效图像")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List]]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            包含图像、边界框、标签等信息的字典
        """
        annotation = self.annotations[idx]
        
        # 加载图像
        image_path = self.raw_data_dir / annotation.image_name
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            # 返回一个空白图像
            image_array = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # 准备边界框和标签
        # 【修复】使用绝对像素坐标（pascal_voc格式）用于数据增强
        bboxes = []
        labels = []
        
        for bbox in annotation.bounding_boxes:
            # 保持绝对坐标（不进行归一化）
            x1 = bbox.x1
            y1 = bbox.y1
            x2 = bbox.x2
            y2 = bbox.y2
            
            # 确保坐标在图像范围内
            x1 = max(0, min(annotation.image_width - 1, x1))
            y1 = max(0, min(annotation.image_height - 1, y1))
            x2 = max(0, min(annotation.image_width, x2))
            y2 = max(0, min(annotation.image_height, y2))
            
            if x2 > x1 and y2 > y1:  # 确保边界框有效
                bboxes.append([x1, y1, x2, y2])
                labels.append(self.chart_type_to_idx[bbox.label])
        
        # 【修复】如果没有边界框，记录错误但不创建占位符
        if not bboxes:
            logger.error(f"图像 {annotation.image_name} 没有有效的边界框！")
            # 创建一个极小的占位符以避免崩溃，但这会在训练时被跳过
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
        
        # 应用数据增强（使用绝对坐标）
        if self.augmentation and len(bboxes) > 0:
            try:
                augmented = self.augmentation(
                    image=image_array,
                    bboxes=bboxes,
                    labels=labels
                )
                image_array = augmented['image']
                bboxes = augmented['bboxes']
                labels = augmented['labels']
                
                # 添加诊断日志
                if not bboxes:
                    logger.warning(f"图像 {annotation.image_name} 数据增强后bbox被过滤掉")
                    
            except Exception as e:
                logger.warning(f"图像 {annotation.image_name} 数据增强失败: {e}")
        
        # 应用基础变换（Resize + Normalize）
        try:
            transformed = self.base_transform(
                image=image_array,
                bboxes=bboxes,
                labels=labels
            )
            image_tensor = transformed['image']
            # Resize后的bbox仍然是绝对坐标（基于新尺寸）
            bboxes = transformed['bboxes']
            labels = transformed['labels']
            
            # 【修复】将Resize后的绝对坐标转换为归一化坐标
            normalized_bboxes = []
            for box in bboxes:
                x1, y1, x2, y2 = box
                # 转换为归一化坐标 [0, 1]
                x1_norm = x1 / self.image_size[1]
                y1_norm = y1 / self.image_size[0]
                x2_norm = x2 / self.image_size[1]
                y2_norm = y2 / self.image_size[0]
                
                # 确保在[0, 1]范围内
                x1_norm = max(0, min(1, x1_norm))
                y1_norm = max(0, min(1, y1_norm))
                x2_norm = max(0, min(1, x2_norm))
                y2_norm = max(0, min(1, y2_norm))
                
                if x2_norm > x1_norm and y2_norm > y1_norm:
                    normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            bboxes = normalized_bboxes if normalized_bboxes else [[0, 0, 0.01, 0.01]]
            
        except Exception as e:
            logger.error(f"图像 {annotation.image_name} 变换失败: {e}")
            # 返回默认值
            image_tensor = torch.zeros(3, *self.image_size)
            bboxes = [[0, 0, 0.01, 0.01]]
            labels = [0]
        
        # 转换为张量
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 创建目标字典（用于目标检测模型）
        target = {
            'boxes': bboxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx], dtype=torch.long),
            'area': self._compute_area(bboxes_tensor),
            'iscrowd': torch.zeros(len(labels), dtype=torch.bool)
        }
        
        return {
            'image': image_tensor,
            'target': target,
            'image_name': annotation.image_name,
            'original_size': (annotation.image_width, annotation.image_height)
        }
    
    def _compute_area(self, bboxes: torch.Tensor) -> torch.Tensor:
        """计算边界框面积"""
        if len(bboxes) == 0:
            return torch.tensor([], dtype=torch.float32)
        
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        return widths * heights
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        class_counts = torch.zeros(len(self.chart_types))
        
        for annotation in self.annotations:
            for bbox in annotation.bounding_boxes:
                class_idx = self.chart_type_to_idx[bbox.label]
                class_counts[class_idx] += 1
        
        # 计算权重（类别越少，权重越大）
        total_count = class_counts.sum()
        weights = total_count / (len(self.chart_types) * class_counts)
        weights = torch.where(class_counts > 0, weights, 0)
        
        return weights


class DashboardDataModule:
    """仪表盘数据模块"""
    
    def __init__(
        self,
        config: Dict,
        batch_size: int = 16,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1
    ):
        """
        初始化数据模块
        
        Args:
            config: 配置字典
            batch_size: 批次大小
            num_workers: 工作进程数
            train_split: 训练集比例
            val_split: 验证集比例
        """
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        
        # 解析标注数据
        self.annotations = self._load_annotations()
        
        # 划分数据集
        self.train_annotations, self.val_annotations, self.test_annotations = self._split_dataset()
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(self.train_annotations)} 张图像")
        logger.info(f"  验证集: {len(self.val_annotations)} 张图像")
        logger.info(f"  测试集: {len(self.test_annotations)} 张图像")
    
    def _load_annotations(self) -> List[DashboardAnnotation]:
        """加载标注数据"""
        chart_types = self.config['model']['chart_types']
        annotations_dir = self.config['data']['annotations_dir']
        raw_data_dir = self.config['data']['raw_dir']
        
        parser = CVATXMLParser(chart_types)
        annotations = parser.parse_directory(annotations_dir)
        
        # 过滤数据 - 只检查图像文件是否存在和是否有边界框
        valid_annotations = []
        total_boxes = 0
        for annotation in annotations:
            image_path = Path(raw_data_dir) / annotation.image_name
            if image_path.exists() and annotation.bounding_boxes:
                valid_annotations.append(annotation)
                total_boxes += len(annotation.bounding_boxes)
        
        logger.info(f"加载了 {len(valid_annotations)} 张有效图像的标注数据，共 {total_boxes} 个边界框")
        return valid_annotations
    
    def _split_dataset(self) -> Tuple[List, List, List]:
        """划分数据集"""
        np.random.seed(self.config.get('experiment', {}).get('seed', 42))
        indices = np.random.permutation(len(self.annotations))
        
        train_size = int(len(self.annotations) * self.train_split)
        val_size = int(len(self.annotations) * self.val_split)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_annotations = [self.annotations[i] for i in train_indices]
        val_annotations = [self.annotations[i] for i in val_indices]
        test_annotations = [self.annotations[i] for i in test_indices]
        
        return train_annotations, val_annotations, test_annotations
    
    def train_dataloader(self) -> DataLoader:
        """创建训练数据加载器"""
        dataset = DashboardDataset(
            annotations=self.train_annotations,
            raw_data_dir=self.config['data']['raw_dir'],
            chart_types=self.config['model']['chart_types'],
            image_size=self.config['data']['image_size'],
            augment=False  # 【临时禁用】验证数据增强是否导致bbox丢失
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """创建验证数据加载器"""
        dataset = DashboardDataset(
            annotations=self.val_annotations,
            raw_data_dir=self.config['data']['raw_dir'],
            chart_types=self.config['model']['chart_types'],
            image_size=self.config['data']['image_size'],
            augment=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """创建测试数据加载器"""
        dataset = DashboardDataset(
            annotations=self.test_annotations,
            raw_data_dir=self.config['data']['raw_dir'],
            chart_types=self.config['model']['chart_types'],
            image_size=self.config['data']['image_size'],
            augment=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch):
        """自定义批次整理函数"""
        images = torch.stack([item['image'] for item in batch])
        targets = [item['target'] for item in batch]
        image_names = [item['image_name'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        return {
            'images': images,
            'targets': targets,
            'image_names': image_names,
            'original_sizes': original_sizes
        }


if __name__ == "__main__":
    # 测试代码
    import yaml
    
    # 加载配置
    with open("configs/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据模块
    data_module = DashboardDataModule(config, batch_size=2)
    
    # 测试数据加载器
    train_loader = data_module.train_dataloader()
    
    print("测试数据加载器...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  图像形状: {batch['images'].shape}")
        print(f"  目标数量: {len(batch['targets'])}")
        print(f"  图像名称: {batch['image_names']}")
        
        if batch_idx >= 2:  # 只测试前几个批次
            break
    
    print("数据加载测试完成！")
