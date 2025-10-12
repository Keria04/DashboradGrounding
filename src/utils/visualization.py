"""
# -*- coding: utf-8 -*-
可视化工具模块
用于可视化仪表盘检测和分割结果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DashboardVisualizer:
    """仪表盘可视化器"""
    
    def __init__(self, config: dict):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色配置
        self.colors = config.get('visualization', {}).get('colors', [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ])
        
        # 图表类型颜色映射
        self.chart_types = config['model']['chart_types']
        self.chart_type_colors = {
            chart_type: self.colors[i % len(self.colors)]
            for i, chart_type in enumerate(self.chart_types)
        }
        
        logger.info("可视化器初始化完成")
    
    def visualize_detection(
        self,
        image_path: str,
        boxes: np.ndarray,
        labels: List[str],
        scores: np.ndarray,
        save_path: str = None,
        show_labels: bool = True,
        show_scores: bool = True
    ) -> None:
        """
        可视化检测结果
        
        Args:
            image_path: 图像路径
            boxes: 边界框 [N, 4] (x1, y1, x2, y2)
            labels: 标签列表
            scores: 置信度分数
            save_path: 保存路径
            show_labels: 是否显示标签
            show_scores: 是否显示置信度
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image_array)
        
        # 绘制边界框
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            
            # 获取颜色
            color = self.chart_type_colors.get(label, '#FF0000')
            
            # 创建边界框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            if show_labels or show_scores:
                text = ""
                if show_labels:
                    text += label
                if show_scores:
                    if text:
                        text += f" ({score:.2f})"
                    else:
                        text = f"{score:.2f}"
                
                ax.text(
                    x1, y1 - 5, text,
                    fontsize=10, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                )
        
        ax.set_title(f"仪表盘检测结果 (检测到 {len(boxes)} 个图表)", fontsize=16)
        ax.axis('off')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_segmentation(
        self,
        image_path: str,
        segmentation_mask: np.ndarray,
        class_names: List[str],
        save_path: str = None,
        alpha: float = 0.5
    ) -> None:
        """
        可视化分割结果
        
        Args:
            image_path: 图像路径
            segmentation_mask: 分割掩码 [H, W]
            class_names: 类别名称列表
            save_path: 保存路径
            alpha: 透明度
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # 创建颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 原始图像
        ax1.imshow(image_array)
        ax1.set_title("原始图像", fontsize=14)
        ax1.axis('off')
        
        # 分割结果
        ax2.imshow(image_array)
        
        # 叠加分割掩码
        for class_id, class_name in enumerate(class_names):
            mask = segmentation_mask == class_id
            if mask.any():
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask] = [*colors[class_id], alpha]
                ax2.imshow(colored_mask)
        
        ax2.set_title("分割结果", fontsize=14)
        ax2.axis('off')
        
        # 添加图例
        legend_elements = [
            patches.Patch(color=colors[i], label=class_name)
            for i, class_name in enumerate(class_names)
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分割可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_comparison(
        self,
        image_path: str,
        ground_truth: Dict,
        predictions: Dict,
        save_path: str = None
    ) -> None:
        """
        可视化预测结果与真实标签的对比
        
        Args:
            image_path: 图像路径
            ground_truth: 真实标签
            predictions: 预测结果
            save_path: 保存路径
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 真实标签
        ax1.imshow(image_array)
        if 'boxes' in ground_truth:
            for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
                x1, y1, x2, y2 = box
                color = self.chart_type_colors.get(label, '#00FF00')
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax1.add_patch(rect)
                ax1.text(x1, y1 - 5, label, fontsize=10, color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax1.set_title("真实标签", fontsize=14)
        ax1.axis('off')
        
        # 预测结果
        ax2.imshow(image_array)
        if 'boxes' in predictions:
            for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
                x1, y1, x2, y2 = box
                color = self.chart_type_colors.get(label, '#FF0000')
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax2.add_patch(rect)
                ax2.text(x1, y1 - 5, f"{label} ({score:.2f})", fontsize=10, color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_title("预测结果", fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"对比可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Dict[str, List[float]] = None,
        val_metrics: Dict[str, List[float]] = None,
        save_path: str = None
    ) -> None:
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失
            val_losses: 验证损失
            train_metrics: 训练指标
            val_metrics: 验证指标
            save_path: 保存路径
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 创建子图
        num_plots = 1
        if train_metrics or val_metrics:
            num_plots += len(train_metrics or val_metrics)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        
        # 绘制损失曲线
        axes[0].plot(epochs, train_losses, 'b-', label='训练损失')
        axes[0].plot(epochs, val_losses, 'r-', label='验证损失')
        axes[0].set_title('训练和验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制指标曲线
        plot_idx = 1
        if train_metrics:
            for metric_name, values in train_metrics.items():
                if plot_idx < len(axes):
                    axes[plot_idx].plot(epochs, values, 'b-', label=f'训练{metric_name}')
                    if val_metrics and metric_name in val_metrics:
                        axes[plot_idx].plot(epochs, val_metrics[metric_name], 'r-', label=f'验证{metric_name}')
                    axes[plot_idx].set_title(f'{metric_name}')
                    axes[plot_idx].set_xlabel('Epoch')
                    axes[plot_idx].set_ylabel(metric_name)
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True)
                    plot_idx += 1
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_path: str = None
    ) -> None:
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 10))
        
        # 使用seaborn绘制混淆矩阵
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_dashboard_overview(
        self,
        results: List[Dict],
        save_path: str = None,
        max_images: int = 9
    ) -> None:
        """
        创建仪表盘检测结果概览
        
        Args:
            results: 检测结果列表
            save_path: 保存路径
            max_images: 最大显示图像数
        """
        num_images = min(len(results), max_images)
        cols = 3
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_images):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            result = results[i]
            
            # 加载图像
            image = Image.open(result['image_path']).convert('RGB')
            ax.imshow(image)
            
            # 绘制检测结果
            for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                x1, y1, x2, y2 = box
                color = self.chart_type_colors.get(label, '#FF0000')
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{label} ({score:.2f})", fontsize=8, color=color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            ax.set_title(f"检测到 {len(result['boxes'])} 个图表", fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('仪表盘检测结果概览', fontsize=16)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"概览图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # 测试代码
    import yaml
    
    # 加载配置
    with open("configs/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建可视化器
    visualizer = DashboardVisualizer(config)
    
    print("可视化器测试完成！")
