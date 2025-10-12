"""
# -*- coding: utf-8 -*-
评估指标模块
用于计算目标检测和分类模型的评估指标
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DetectionMetrics:
    """目标检测评估指标"""
    
    def __init__(self, class_names: List[str], iou_threshold: float = 0.5):
        """
        初始化评估指标
        
        Args:
            class_names: 类别名称列表
            iou_threshold: IoU阈值
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_threshold = iou_threshold
        
        logger.info(f"初始化检测指标，类别数: {self.num_classes}")
    
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        计算两个边界框的IoU
        
        Args:
            box1: 边界框1 [x1, y1, x2, y2]
            box2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        # 计算交集
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def match_predictions_targets(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pred_scores: torch.Tensor,
        iou_threshold: float = None
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        匹配预测和真实标签
        
        Args:
            predictions: 预测边界框 [N, 4]
            targets: 真实边界框 [M, 4]
            pred_scores: 预测置信度 [N]
            iou_threshold: IoU阈值
            
        Returns:
            (匹配的预测索引, 匹配的目标索引, IoU值)
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        if len(predictions) == 0 or len(targets) == 0:
            return [], [], []
        
        # 计算IoU矩阵
        ious = torch.zeros(len(predictions), len(targets))
        for i, pred in enumerate(predictions):
            for j, target in enumerate(targets):
                ious[i, j] = self.compute_iou(pred, target)
        
        # 贪心匹配
        matched_preds = []
        matched_targets = []
        matched_ious = []
        
        # 按置信度排序预测
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for pred_idx in sorted_indices:
            pred_idx = pred_idx.item()
            
            # 找到最佳匹配的目标
            best_iou = 0
            best_target_idx = -1
            
            for target_idx in range(len(targets)):
                if target_idx in matched_targets:
                    continue
                
                iou = ious[pred_idx, target_idx]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_target_idx = target_idx
            
            if best_target_idx != -1:
                matched_preds.append(pred_idx)
                matched_targets.append(best_target_idx)
                matched_ious.append(best_iou.item())
        
        return matched_preds, matched_targets, matched_ious
    
    def compute_ap(self, predictions: List[Dict], targets: List[Dict], class_id: int) -> float:
        """
        计算单个类别的AP
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            class_id: 类别ID
            
        Returns:
            AP值
        """
        # 收集该类别的所有预测和真实标签
        class_predictions = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # 过滤该类别的预测
            pred_mask = pred['labels'] == class_id
            if pred_mask.sum() > 0:
                class_predictions.append({
                    'boxes': pred['boxes'][pred_mask],
                    'scores': pred['scores'][pred_mask]
                })
            else:
                class_predictions.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0)
                })
            
            # 过滤该类别的真实标签
            target_mask = target['labels'] == class_id
            if target_mask.sum() > 0:
                class_targets.append({
                    'boxes': target['boxes'][target_mask]
                })
            else:
                class_targets.append({
                    'boxes': torch.empty(0, 4)
                })
        
        # 计算AP
        if len(class_predictions) == 0:
            return 0.0
        
        # 收集所有预测和对应的真实标签数量
        all_scores = []
        all_tp = []
        all_fp = []
        num_targets = 0
        
        for pred, target in zip(class_predictions, class_targets):
            if len(pred['boxes']) == 0:
                num_targets += len(target['boxes'])
                continue
            
            # 匹配预测和真实标签
            matched_preds, matched_targets, matched_ious = self.match_predictions_targets(
                pred['boxes'], target['boxes'], pred['scores']
            )
            
            # 标记TP和FP
            tp = [False] * len(pred['boxes'])
            fp = [False] * len(pred['boxes'])
            
            for i in range(len(pred['boxes'])):
                if i in matched_preds:
                    tp[i] = True
                else:
                    fp[i] = True
            
            all_scores.extend(pred['scores'].tolist())
            all_tp.extend(tp)
            all_fp.extend(fp)
            num_targets += len(target['boxes'])
        
        if len(all_scores) == 0:
            return 0.0
        
        # 按置信度排序
        sorted_indices = np.argsort(all_scores)[::-1]
        tp = np.array(all_tp)[sorted_indices]
        fp = np.array(all_fp)[sorted_indices]
        
        # 计算精确率和召回率
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / max(num_targets, 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 计算AP (使用11点插值)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def compute_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 计算每个类别的AP
        aps = []
        for class_id in range(self.num_classes):
            ap = self.compute_ap(predictions, targets, class_id)
            aps.append(ap)
            metrics[f'AP_{self.class_names[class_id]}'] = ap
        
        # 计算mAP
        mAP = np.mean(aps) if aps else 0.0
        metrics['mAP'] = mAP
        
        # 计算整体统计信息
        total_predictions = sum(len(pred['boxes']) for pred in predictions)
        total_targets = sum(len(target['boxes']) for target in targets)
        
        metrics['total_predictions'] = total_predictions
        metrics['total_targets'] = total_targets
        
        # 计算平均置信度
        all_scores = []
        for pred in predictions:
            if len(pred['scores']) > 0:
                all_scores.extend(pred['scores'].tolist())
        
        metrics['avg_confidence'] = np.mean(all_scores) if all_scores else 0.0
        
        return metrics


class ClassificationMetrics:
    """分类评估指标"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化分类指标
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def compute_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            predictions: 预测标签 [N]
            targets: 真实标签 [N]
            
        Returns:
            混淆矩阵
        """
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for pred, target in zip(predictions, targets):
            matrix[target, pred] += 1
        
        return matrix
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            predictions: 预测标签
            targets: 真实标签
            
        Returns:
            评估指标字典
        """
        # 计算准确率
        accuracy = (predictions == targets).float().mean().item()
        
        # 计算混淆矩阵
        confusion_matrix = self.compute_confusion_matrix(predictions, targets)
        
        # 计算每个类别的精确率、召回率和F1分数
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(self.num_classes):
            # 精确率
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # 召回率
            fn = confusion_matrix[i, :].sum() - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # 计算宏平均和微平均
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)
        
        # 构建指标字典
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
        
        # 添加每个类别的指标
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precisions[i]
            metrics[f'recall_{class_name}'] = recalls[i]
            metrics[f'f1_{class_name}'] = f1_scores[i]
        
        return metrics


def compute_bbox_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    计算边界框相关的评估指标
    
    Args:
        predictions: 预测结果
        targets: 真实标签
        
    Returns:
        边界框评估指标
    """
    metrics = DetectionMetrics(['dummy'])  # 临时类别名
    
    # 计算IoU分布
    ious = []
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
            for pred_box in pred['boxes']:
                for target_box in target['boxes']:
                    iou = metrics.compute_iou(pred_box, target_box)
                    ious.append(iou.item())
    
    if not ious:
        return {'avg_iou': 0.0, 'median_iou': 0.0}
    
    return {
        'avg_iou': np.mean(ious),
        'median_iou': np.median(ious),
        'min_iou': np.min(ious),
        'max_iou': np.max(ious)
    }


if __name__ == "__main__":
    # 测试代码
    import torch
    
    # 测试检测指标
    class_names = ['Bar Chart', 'Line Chart', 'Pie Chart']
    metrics = DetectionMetrics(class_names)
    
    # 创建测试数据
    predictions = [
        {
            'boxes': torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),
            'labels': torch.tensor([0, 1]),
            'scores': torch.tensor([0.9, 0.8])
        }
    ]
    
    targets = [
        {
            'boxes': torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),
            'labels': torch.tensor([0, 1])
        }
    ]
    
    # 计算指标
    results = metrics.compute_metrics(predictions, targets)
    print("检测指标测试结果:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n测试完成！")
