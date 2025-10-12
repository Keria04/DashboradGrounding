"""
# -*- coding: utf-8 -*-
XML标注文件解析器
用于解析CVAT标注的XML文件，提取图表区域的边界框和类型信息
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import os
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """边界框数据结构"""
    x1: float  # 左上角x坐标
    y1: float  # 左上角y坐标
    x2: float  # 右下角x坐标
    y2: float  # 右下角y坐标
    label: str  # 图表类型标签
    confidence: float = 1.0  # 置信度（标注数据默认为1.0）
    
    @property
    def width(self) -> float:
        """边界框宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """边界框高度"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """边界框面积"""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """边界框中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DashboardAnnotation:
    """仪表盘标注数据结构"""
    image_name: str
    image_width: int
    image_height: int
    bounding_boxes: List[BoundingBox]


class CVATXMLParser:
    """CVAT XML标注文件解析器"""
    
    def __init__(self, chart_types: List[str]):
        """
        初始化解析器
        
        Args:
            chart_types: 支持的图表类型列表
        """
        self.chart_types = chart_types
        self.chart_type_mapping = {chart_type.lower(): chart_type for chart_type in chart_types}
    
    def parse_xml_file(self, xml_path: str) -> List[DashboardAnnotation]:
        """
        解析XML标注文件
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            DashboardAnnotation列表
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotations = []
            
            # 遍历所有图像
            for image_elem in root.findall('image'):
                annotation = self._parse_image_element(image_elem)
                if annotation:
                    annotations.append(annotation)
            
            logger.info(f"成功解析 {xml_path}，共 {len(annotations)} 张图像")
            return annotations
            
        except ET.ParseError as e:
            logger.error(f"XML解析错误 {xml_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"解析文件时发生错误 {xml_path}: {e}")
            return []
    
    def _parse_image_element(self, image_elem: ET.Element) -> Optional[DashboardAnnotation]:
        """
        解析单个图像元素
        
        Args:
            image_elem: 图像XML元素
            
        Returns:
            DashboardAnnotation对象或None
        """
        try:
            # 获取图像基本信息
            image_name = image_elem.get('name', '')
            image_width = int(image_elem.get('width', 0))
            image_height = int(image_elem.get('height', 0))
            
            if not image_name or image_width <= 0 or image_height <= 0:
                logger.warning(f"图像信息不完整: {image_name}")
                return None
            
            bounding_boxes = []
            
            # 解析边界框
            for box_elem in image_elem.findall('box'):
                bbox = self._parse_box_element(box_elem)
                if bbox:
                    bounding_boxes.append(bbox)
            
            if not bounding_boxes:
                logger.warning(f"图像 {image_name} 没有有效的边界框")
                return None
            
            return DashboardAnnotation(
                image_name=image_name,
                image_width=image_width,
                image_height=image_height,
                bounding_boxes=bounding_boxes
            )
            
        except Exception as e:
            logger.error(f"解析图像元素时发生错误: {e}")
            return None
    
    def _parse_box_element(self, box_elem: ET.Element) -> Optional[BoundingBox]:
        """
        解析边界框元素
        
        Args:
            box_elem: 边界框XML元素
            
        Returns:
            BoundingBox对象或None
        """
        try:
            # 获取边界框坐标
            x1 = float(box_elem.get('xtl', 0))
            y1 = float(box_elem.get('ytl', 0))
            x2 = float(box_elem.get('xbr', 0))
            y2 = float(box_elem.get('ybr', 0))
            
            # 获取标签
            label = box_elem.get('label', '').strip()
            
            # 验证数据
            if not label or x1 >= x2 or y1 >= y2:
                logger.warning(f"边界框数据无效: label={label}, bbox=({x1}, {y1}, {x2}, {y2})")
                return None
            
            # 标准化标签名称
            normalized_label = self._normalize_label(label)
            # 【修复】_normalize_label已经确保返回的标签在chart_types中
            # 如果返回原始标签说明没匹配到，需要进一步检查
            if normalized_label not in self.chart_types:
                # 尝试再次匹配（处理_normalize_label可能的bug）
                for chart_type in self.chart_types:
                    if normalized_label.lower() == chart_type.lower():
                        normalized_label = chart_type
                        break
                else:
                    logger.warning(f"未知的图表类型: '{label}' -> '{normalized_label}'")
                    logger.warning(f"  可用类型: {self.chart_types[:3]}...")
                    return None
            
            return BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                label=normalized_label
            )
            
        except (ValueError, TypeError) as e:
            logger.error(f"解析边界框时发生错误: {e}")
            return None
    
    def _normalize_label(self, label: str) -> str:
        """
        标准化标签名称
        
        Args:
            label: 原始标签
            
        Returns:
            标准化后的标签
        """
        # 去除首尾空格
        normalized = label.strip()
        
        # 直接精确匹配
        if normalized in self.chart_types:
            return normalized
        
        # 忽略大小写匹配
        for chart_type in self.chart_types:
            if normalized.lower() == chart_type.lower():
                return chart_type
        
        # 修复常见拼写错误
        spelling_fixes = {
            'calender': 'calendar',
            'rader': 'radar',
            'heatmap': 'heatmap',
        }
        
        lower_normalized = normalized.lower()
        for wrong, correct in spelling_fixes.items():
            if wrong in lower_normalized:
                lower_normalized = lower_normalized.replace(wrong, correct)
                # 再次尝试匹配
                for chart_type in self.chart_types:
                    if lower_normalized == chart_type.lower():
                        return chart_type
        
        # 部分匹配 (最后的选择)
        for standard_label in self.chart_types:
            if normalized.lower() in standard_label.lower() or standard_label.lower() in normalized.lower():
                return standard_label
        
        return label  # 如果无法匹配，返回原始标签
    
    def parse_directory(self, annotations_dir: str) -> List[DashboardAnnotation]:
        """
        解析整个标注目录
        
        Args:
            annotations_dir: 标注目录路径
            
        Returns:
            所有标注数据的列表
        """
        all_annotations = []
        
        if not os.path.exists(annotations_dir):
            logger.error(f"标注目录不存在: {annotations_dir}")
            return all_annotations
        
        # 遍历所有标注者目录
        for annotator_dir in os.listdir(annotations_dir):
            annotator_path = os.path.join(annotations_dir, annotator_dir)
            
            if not os.path.isdir(annotator_path):
                continue
            
            # 查找XML文件
            for file_name in os.listdir(annotator_path):
                if file_name.endswith('.xml'):
                    xml_path = os.path.join(annotator_path, file_name)
                    annotations = self.parse_xml_file(xml_path)
                    all_annotations.extend(annotations)
        
        logger.info(f"总共解析了 {len(all_annotations)} 张图像的标注数据")
        return all_annotations
    
    def get_statistics(self, annotations: List[DashboardAnnotation]) -> Dict:
        """
        获取标注数据统计信息
        
        Args:
            annotations: 标注数据列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_images': len(annotations),
            'total_boxes': sum(len(ann.bounding_boxes) for ann in annotations),
            'chart_type_counts': {},
            'image_size_distribution': {},
            'box_size_distribution': {'small': 0, 'medium': 0, 'large': 0}
        }
        
        # 统计图表类型
        for annotation in annotations:
            for bbox in annotation.bounding_boxes:
                chart_type = bbox.label
                stats['chart_type_counts'][chart_type] = stats['chart_type_counts'].get(chart_type, 0) + 1
        
        # 统计图像尺寸分布
        for annotation in annotations:
            size_key = f"{annotation.image_width}x{annotation.image_height}"
            stats['image_size_distribution'][size_key] = stats['image_size_distribution'].get(size_key, 0) + 1
        
        # 统计边界框尺寸分布
        all_areas = []
        for annotation in annotations:
            for bbox in annotation.bounding_boxes:
                all_areas.append(bbox.area)
        
        if all_areas:
            avg_area = sum(all_areas) / len(all_areas)
            for area in all_areas:
                if area < avg_area * 0.5:
                    stats['box_size_distribution']['small'] += 1
                elif area < avg_area * 1.5:
                    stats['box_size_distribution']['medium'] += 1
                else:
                    stats['box_size_distribution']['large'] += 1
        
        return stats


def load_chart_types_from_config(config_path: str) -> List[str]:
    """
    从配置文件加载图表类型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        图表类型列表
    """
    import yaml
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('model', {}).get('chart_types', [])
    
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return []


if __name__ == "__main__":
    # 测试代码
    import yaml
    
    # 加载配置
    config_path = "configs/config.yaml"
    chart_types = load_chart_types_from_config(config_path)
    
    # 创建解析器
    parser = CVATXMLParser(chart_types)
    
    # 解析标注数据
    annotations_dir = "data/annotations"
    annotations = parser.parse_directory(annotations_dir)
    
    # 打印统计信息
    stats = parser.get_statistics(annotations)
    print("标注数据统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
