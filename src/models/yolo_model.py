"""
# -*- coding: utf-8 -*-
YOLO模型实现
基于数据分析结果，实现专门针对仪表盘图表检测的YOLO模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops
from torch.utils.data import DataLoader
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class YOLOv8Backbone(nn.Module):
    """YOLOv8骨干网络"""
    
    def __init__(self, input_channels=3, num_classes=20):
        super(YOLOv8Backbone, self).__init__()
        
        # CSPDarknet骨干网络
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1),  # 640 -> 320
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        
        # Stage 1
        self.stage1 = self._make_csp_layer(32, 64, 1)
        self.stage2 = self._make_csp_layer(64, 128, 2)
        self.stage3 = self._make_csp_layer(128, 256, 2)
        self.stage4 = self._make_csp_layer(256, 512, 2)
        self.stage5 = self._make_csp_layer(512, 1024, 2)
        
        # SPPF
        self.sppf = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(5, 1, 2),
            nn.MaxPool2d(5, 1, 2),
            nn.MaxPool2d(5, 1, 2),
            nn.Conv2d(512, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),
        )
        
    def _make_csp_layer(self, in_channels, out_channels, num_blocks):
        """创建CSP层"""
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            CSPBlock(out_channels, num_blocks)
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)      # 640 -> 320
        x = self.stage1(x)    # 320 -> 160
        x = self.stage2(x)    # 160 -> 80
        x = self.stage3(x)    # 80 -> 40
        x = self.stage4(x)    # 40 -> 20
        x = self.stage5(x)    # 20 -> 10
        x = self.sppf(x)      # 10 -> 10
        return x


class CSPBlock(nn.Module):
    """CSP块"""
    
    def __init__(self, channels, num_blocks):
        super(CSPBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels // 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(channels, channels // 2, 1, 1, 0)
        
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels // 2) for _ in range(num_blocks)
        ])
        
        self.conv3 = nn.Conv2d(channels, channels, 1, 1, 0)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        x2 = self.blocks(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = x + residual
        x = self.activation(x)
        
        return x


class YOLOv8Neck(nn.Module):
    """YOLOv8颈部网络"""
    
    def __init__(self, num_classes=20):
        super(YOLOv8Neck, self).__init__()
        
        # 上采样路径
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)
        
        # 特征融合
        self.conv1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv2 = nn.Conv2d(512 + 512, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 512, 1, 1, 0)
        
        self.conv4 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv5 = nn.Conv2d(256 + 256, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 1, 1, 0)
        
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0)
        self.conv8 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.conv9 = nn.Conv2d(128, 128, 1, 1, 0)
        
        # 检测头
        self.detect1 = nn.Conv2d(128, 64 + num_classes + 4, 1, 1, 0)  # 80x80
        self.detect2 = nn.Conv2d(256, 64 + num_classes + 4, 1, 1, 0)  # 40x40
        self.detect3 = nn.Conv2d(512, 64 + num_classes + 4, 1, 1, 0)  # 20x20
        
    def forward(self, x):
        # x shape: [B, 1024, 20, 20]
        
        # P5 -> P4
        p5 = self.upsample1(x)  # [B, 1024, 40, 40]
        p5 = self.conv1(p5)     # [B, 512, 40, 40]
        
        # 假设有来自骨干网络的512维特征
        # p4 = torch.cat([p5, features_512], dim=1)  # [B, 1024, 40, 40]
        # 简化处理，使用p5作为p4
        p4 = torch.cat([p5, p5], dim=1)  # [B, 1024, 40, 40]
        p4 = self.conv2(p4)              # [B, 512, 40, 40]
        p4 = self.conv3(p4)              # [B, 512, 40, 40]
        
        # P4 -> P3
        p4_up = self.upsample2(p4)       # [B, 512, 80, 80]
        p4_up = self.conv4(p4_up)        # [B, 256, 80, 80]
        
        # 假设有来自骨干网络的256维特征
        # p3 = torch.cat([p4_up, features_256], dim=1)  # [B, 512, 80, 80]
        p3 = torch.cat([p4_up, p4_up], dim=1)  # [B, 512, 80, 80]
        p3 = self.conv5(p3)                     # [B, 256, 80, 80]
        p3 = self.conv6(p3)                     # [B, 256, 80, 80]
        
        # P3 -> P2
        p3_up = self.upsample3(p3)       # [B, 256, 160, 160]
        p3_up = self.conv7(p3_up)        # [B, 128, 160, 160]
        
        # 假设有来自骨干网络的128维特征
        # p2 = torch.cat([p3_up, features_128], dim=1)  # [B, 256, 160, 160]
        p2 = torch.cat([p3_up, p3_up], dim=1)  # [B, 256, 160, 160]
        p2 = self.conv8(p2)                     # [B, 128, 160, 160]
        p2 = self.conv9(p2)                     # [B, 128, 160, 160]
        
        # 检测头
        det1 = self.detect1(p2)  # [B, 64+20+4, 160, 160] - 小目标
        det2 = self.detect2(p3)  # [B, 64+20+4, 80, 80]  - 中等目标
        det3 = self.detect3(p4)  # [B, 64+20+4, 40, 40]  - 大目标
        
        return [det1, det2, det3]


class YOLOv8Head(nn.Module):
    """YOLOv8检测头"""
    
    def __init__(self, num_classes=20, num_anchors=1):
        super(YOLOv8Head, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def forward(self, features):
        """
        features: List of tensors [det1, det2, det3]
        det shape: [B, 64+20+4, H, W]
        """
        outputs = []
        
        for det in features:
            B, C, H, W = det.shape
            
            # 重塑为 [B, H, W, 64+20+4]
            det = det.permute(0, 2, 3, 1).contiguous()
            
            # 分离不同组件
            # 假设前64维是特征，接下来20维是类别，最后4维是边界框
            features_part = det[..., :64]  # [B, H, W, 64]
            cls_part = det[..., 64:64+self.num_classes]  # [B, H, W, 20]
            box_part = det[..., 64+self.num_classes:]  # [B, H, W, 4]
            
            # 应用sigmoid到类别和边界框
            cls_part = torch.sigmoid(cls_part)
            box_part = torch.sigmoid(box_part)
            
            outputs.append({
                'features': features_part,
                'class': cls_part,
                'box': box_part,
                'grid_size': (H, W)
            })
        
        return outputs


class DashboardYOLO(nn.Module):
    """仪表盘YOLO模型"""
    
    def __init__(self, num_classes=20, input_size=640):
        super(DashboardYOLO, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # 骨干网络
        self.backbone = YOLOv8Backbone(input_channels=3, num_classes=num_classes)
        
        # 颈部网络
        self.neck = YOLOv8Neck(num_classes=num_classes)
        
        # 检测头
        self.head = YOLOv8Head(num_classes=num_classes)
        
        logger.info(f"初始化仪表盘YOLO模型，类别数: {num_classes}, 输入尺寸: {input_size}")
    
    def forward(self, x):
        """前向传播"""
        # 骨干网络特征提取
        features = self.backbone(x)
        
        # 颈部网络特征融合
        neck_features = self.neck(features)
        
        # 检测头预测
        outputs = self.head(neck_features)
        
        return outputs
    
    def predict(self, x, confidence_threshold=0.5, nms_threshold=0.5):
        """预测方法"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(x)
            
            # 后处理
            predictions = self._postprocess(outputs, confidence_threshold, nms_threshold)
            
        return predictions
    
    def _postprocess(self, outputs, confidence_threshold, nms_threshold):
        """后处理"""
        batch_predictions = []
        
        batch_size = outputs[0]['class'].shape[0]
        
        for b in range(batch_size):
            boxes = []
            scores = []
            labels = []
            
            for output in outputs:
                cls_pred = output['class'][b]  # [H, W, num_classes]
                box_pred = output['box'][b]    # [H, W, 4]
                H, W = output['grid_size']
                
                # 获取最高置信度的类别
                max_scores, max_indices = torch.max(cls_pred, dim=-1)
                
                # 过滤低置信度预测
                valid_mask = max_scores > confidence_threshold
                
                if valid_mask.any():
                    valid_scores = max_scores[valid_mask]
                    valid_labels = max_indices[valid_mask]
                    valid_boxes = box_pred[valid_mask]
                    
                    # 转换边界框坐标
                    valid_boxes = self._decode_boxes(valid_boxes, H, W)
                    
                    boxes.extend(valid_boxes.tolist())
                    scores.extend(valid_scores.tolist())
                    labels.extend(valid_labels.tolist())
            
            # 应用NMS
            if boxes:
                boxes = torch.tensor(boxes)
                scores = torch.tensor(scores)
                labels = torch.tensor(labels)
                
                keep_indices = self._apply_nms(boxes, scores, nms_threshold)
                
                batch_predictions.append({
                    'boxes': boxes[keep_indices],
                    'scores': scores[keep_indices],
                    'labels': labels[keep_indices]
                })
            else:
                batch_predictions.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long)
                })
        
        return batch_predictions
    
    def _decode_boxes(self, boxes, H, W):
        """解码边界框坐标"""
        # boxes: [N, 4] in range [0, 1]
        # 转换为绝对坐标
        boxes = boxes * torch.tensor([W, H, W, H], device=boxes.device)
        return boxes
    
    def _apply_nms(self, boxes, scores, nms_threshold):
        """应用非极大值抑制"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # 计算IoU并应用NMS
        keep_indices = torchvision.ops.nms(boxes, scores, nms_threshold)
        return keep_indices


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_yolo_model(config):
    """创建YOLO模型"""
    chart_types = config['model']['chart_types']
    num_classes = len(chart_types)
    input_size = config['model'].get('input_size', [640, 640])[0]
    
    model = DashboardYOLO(num_classes=num_classes, input_size=input_size)
    
    logger.info(f"创建YOLO模型，类别数: {num_classes}, 输入尺寸: {input_size}")
    
    return model


if __name__ == "__main__":
    # 测试代码
    import yaml
    
    # 加载配置
    with open("configs/config_yolo.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = create_yolo_model(config)
    
    # 测试前向传播
    batch_size = 2
    input_size = 640
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    
    print(f"输入形状: {dummy_input.shape}")
    
    # 前向传播
    outputs = model(dummy_input)
    print(f"输出数量: {len(outputs)}")
    
    for i, output in enumerate(outputs):
        print(f"输出 {i}: {output['class'].shape}, {output['box'].shape}")
    
    # 测试预测
    predictions = model.predict(dummy_input)
    print(f"预测结果数量: {len(predictions)}")
    
    print("YOLO模型测试完成！")
