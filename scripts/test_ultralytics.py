#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试官方ultralytics YOLO模型
验证第一阶段的两个任务
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.encoding_config import initialize_encoding
    initialize_encoding()
except:
    pass

from scripts.inference_ultralytics import UltralyticsInference

def test_official_yolo():
    """测试官方YOLO模型"""
    print("="*70)
    print("官方ultralytics YOLO - 第一阶段任务验证")
    print("="*70)
    
    # 查找模型
    model_path = project_root / 'experiments' / 'ultralytics_yolo' / 'weights' / 'best.pt'
    
    if not model_path.exists():
        print(f"\n错误: 模型文件不存在")
        print(f"  路径: {model_path}")
        print(f"\n请先完成训练:")
        print(f"  python scripts\\train_ultralytics_yolo.py")
        return False
    
    print(f"\n找到模型: {model_path}")
    
    # 创建推理器
    print(f"\n初始化推理器...")
    inference = UltralyticsInference(str(model_path))
    
    # 测试图像
    test_image = project_root / 'data' / 'raw' / 'dashboard_0001.png'
    
    if not test_image.exists():
        print(f"\n错误: 测试图像不存在: {test_image}")
        return False
    
    # ===== 任务1测试 =====
    print(f"\n" + "="*70)
    print("任务1: 图表检测与分类")
    print("="*70)
    print(f"测试图像: {test_image.name}")
    
    result = inference.detect_charts(str(test_image), conf=0.25)
    
    print(f"\n检测到 {len(result['boxes'])} 个图表")
    
    if len(result['boxes']) > 0:
        # 统计类型分布
        from collections import Counter
        label_counts = Counter(result['labels'])
        
        print(f"\n类型分布:")
        for label, count in label_counts.most_common():
            print(f"  - {label}: {count}个")
        
        print(f"\n详细信息 (前10个):")
        for i, (label, score) in enumerate(zip(
            result['labels'][:10],
            result['scores'][:10]
        ), 1):
            print(f"  {i}. {label}: {score:.2%}")
        
        # 保存可视化
        output_path = project_root / 'output' / 'ultralytics_task1.png'
        inference.visualize(str(test_image), str(output_path), conf=0.25)
        print(f"\n✓ 可视化已保存: {output_path}")
        
        task1_success = len(set(result['labels'])) > 1  # 至少2种类型
    else:
        print(f"\n未检测到图表")
        task1_success = False
    
    # ===== 任务2测试 =====
    print(f"\n" + "="*70)
    print("任务2: 文本查询定位")
    print("="*70)
    
    queries = ["找到折线图", "找到柱状图", "找到饼图"]
    task2_success = False
    
    for query in queries:
        query_result = inference.find_chart_by_text(
            str(test_image),
            query,
            conf=0.25
        )
        
        print(f"\n查询: '{query}'")
        print(f"  找到 {len(query_result['matched_charts'])} 个匹配")
        
        if len(query_result['matched_charts']) > 0:
            task2_success = True
            for chart in query_result['matched_charts'][:3]:
                print(f"    - {chart['label']} ({chart['score']:.2%})")
    
    # ===== 总结 =====
    print(f"\n" + "="*70)
    print("验证总结")
    print("="*70)
    
    if task1_success:
        print(f"✓ 任务1: 成功 - 能检测并分类多种图表")
    else:
        print(f"✗ 任务2: 失败 - 检测质量不佳")
    
    if task2_success:
        print(f"✓ 任务2: 成功 - 文本查询能找到图表")
    else:
        print(f"✗ 任务2: 失败 - 未找到匹配")
    
    if task1_success and task2_success:
        print(f"\n🎉 第一阶段两个任务全部完成！")
        return True
    else:
        print(f"\n需要继续优化...")
        return False

if __name__ == "__main__":
    success = test_official_yolo()
    sys.exit(0 if success else 1)

