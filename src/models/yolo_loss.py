#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO损失函数
实现YOLOv8风格的损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class YOLOLoss(nn.Module):
    """YOLO损失函数"""
    
    def __init__(self, num_classes=20, input_size=640):
        super(YOLOLoss, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # 损失权重
        self.lambda_box = 7.5
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        
    def forward(self, predictions: List[Dict], targets: List[Dict]):
        """
        计算YOLO损失
        
        Args:
            predictions: 模型输出，包含3个尺度的预测
            targets: 真实标签列表
            
        Returns:
            总损失字典
        """
        device = predictions[0]['box'].device
        batch_size = predictions[0]['box'].shape[0]
        
        total_box_loss = 0
        total_cls_loss = 0
        total_obj_loss = 0
        
        # 统计总的GT box数量用于正确平均
        total_gt_boxes = 0
        
        # 遍历每个样本
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            
            # 获取目标边界框和标签
            if 'boxes' not in target or len(target['boxes']) == 0:
                # 没有目标，跳过（不计算背景损失，避免loss膨胀）
                continue
            
            gt_boxes = target['boxes'].to(device)  # [N, 4] - 归一化坐标
            gt_labels = target['labels'].to(device)  # [N]
            
            num_gt = len(gt_boxes)
            total_gt_boxes += num_gt
            
            # 将归一化坐标转换为像素坐标
            gt_boxes_pixel = gt_boxes.clone()
            gt_boxes_pixel[:, [0, 2]] *= self.input_size
            gt_boxes_pixel[:, [1, 3]] *= self.input_size
            
            # 【修复】只在最合适的尺度计算loss，避免重复
            # 为每个GT box找到最合适的预测尺度
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_pixel, gt_labels)):
                # 计算box的宽高
                box_w = gt_box[2] - gt_box[0]
                box_h = gt_box[3] - gt_box[1]
                box_area = box_w * box_h
                
                # 根据box大小选择最合适的尺度
                # 小目标 -> 大feature map (stride=8)
                # 大目标 -> 小feature map (stride=32)
                if box_area < (self.input_size / 4) ** 2:
                    pred_idx = 0  # 最大的feature map (80x80)
                elif box_area < (self.input_size / 2) ** 2:
                    pred_idx = 1  # 中等feature map (40x40)
                else:
                    pred_idx = 2  # 最小feature map (20x20)
                
                # 只在选定的尺度计算loss
                pred = predictions[pred_idx]
                H, W = pred['grid_size']
                stride = self.input_size / H
                
                cls_pred = pred['class'][batch_idx]  # [H, W, num_classes]
                box_pred = pred['box'][batch_idx]  # [H, W, 4]
                
                # 计算box中心所在的网格
                center_x = (gt_box[0] + gt_box[2]) / 2
                center_y = (gt_box[1] + gt_box[3]) / 2
                
                grid_x = int(center_x / stride)
                grid_y = int(center_y / stride)
                
                # 确保在网格范围内
                grid_x = max(0, min(W - 1, grid_x))
                grid_y = max(0, min(H - 1, grid_y))
                
                # 类别损失
                target_cls = torch.zeros(self.num_classes, device=device)
                target_cls[gt_label] = 1.0
                total_cls_loss += F.binary_cross_entropy(
                    cls_pred[grid_y, grid_x],
                    target_cls
                )
                
                # 边界框损失 (使用IoU loss)
                pred_box = box_pred[grid_y, grid_x]
                
                # 将预测的归一化偏移转换为实际坐标
                pred_x = (grid_x + pred_box[0]) * stride
                pred_y = (grid_y + pred_box[1]) * stride
                pred_w = pred_box[2] * self.input_size
                pred_h = pred_box[3] * self.input_size
                
                pred_x1 = pred_x - pred_w / 2
                pred_y1 = pred_y - pred_h / 2
                pred_x2 = pred_x + pred_w / 2
                pred_y2 = pred_y + pred_h / 2
                
                pred_box_abs = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2])
                
                # 计算IoU loss
                iou_loss = 1.0 - self._calculate_iou(pred_box_abs, gt_box)
                total_box_loss += iou_loss
        
        # 【修复】按GT box数量平均，而不是batch size
        if total_gt_boxes > 0:
            box_loss = total_box_loss / total_gt_boxes
            cls_loss = total_cls_loss / total_gt_boxes
        else:
            box_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)
        
        obj_loss = torch.tensor(0.0, device=device)  # 简化版本暂不计算objectness
        
        # 总损失
        total_loss = (
            self.lambda_box * box_loss +
            self.lambda_cls * cls_loss +
            self.lambda_obj * obj_loss
        )
        
        return {
            'loss': total_loss,
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'obj_loss': obj_loss
        }
    
    def _calculate_iou(self, box1, box2):
        """计算两个box的IoU"""
        # box1: [4] - [x1, y1, x2, y2]
        # box2: [4] - [x1, y1, x2, y2]
        
        # 计算交集
        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou

