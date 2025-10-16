# 📊 数据管理规范与架构设计

## 🎯 目录结构设计

```
data/
├── 📂 raw/                          # 原始数据（只添加，不删除）
│   ├── batch_001/                   # 第一批数据 (已有279张)
│   │   ├── dashboard_0001.png
│   │   ├── dashboard_0002.png
│   │   └── ...
│   │
│   ├── batch_002/                   # 第二批数据 (新增)
│   │   ├── dashboard_0280.png
│   │   ├── dashboard_0281.png
│   │   └── ...
│   │
│   └── batch_XXX/                   # 后续批次...
│
├── 📂 annotations/                  # 标注数据（CVAT XML格式）
│   ├── batch_001/                   # 对应第一批
│   │   ├── annotator1/
│   │   │   └── 0001-0069.xml
│   │   ├── annotator2/
│   │   │   └── 0070-0139.xml
│   │   ├── annotator3/
│   │   │   └── 0140-0209.xml
│   │   └── annotator4/
│   │       └── 0210-0279.xml
│   │
│   ├── batch_002/                   # 对应第二批
│   │   ├── annotator1/
│   │   │   └── 0280-0349.xml
│   │   └── ...
│   │
│   └── batch_XXX/                   # 后续批次...
│
├── 📂 yolo_format/                  # 训练用YOLO格式（自动生成）
│   ├── dashboard.yaml               # YOLO配置文件
│   ├── images/
│   │   ├── train/                   # 训练集图片
│   │   ├── val/                     # 验证集图片
│   │   └── test/                    # 测试集图片
│   └── labels/
│       ├── train/                   # 训练集标签
│       ├── val/                     # 验证集标签
│       └── test/                    # 测试集标签
│
├── 📂 statistics/                   # 数据统计（新增）
│   ├── data_overview.json           # 数据总览
│   ├── class_distribution.json      # 类别分布
│   └── batch_info.json              # 批次信息
│
└── 📄 data_index.csv                # 数据索引表（新增）
```

---

## 📋 添加新数据的标准流程

### Step 1: 准备新数据

#### 1.1 创建新批次目录
```bash
# 确定批次编号（查看现有最大编号+1）
# 假设要添加第二批数据

# 创建目录
mkdir data/raw/batch_002
mkdir data/annotations/batch_002
```

#### 1.2 文件命名规范
```
图片命名: dashboard_XXXX.png  (XXXX为4位数字，从上一批次最后编号+1开始)
标注命名: XXXX-YYYY.xml       (XXXX起始编号, YYYY结束编号)

示例:
- 第一批: dashboard_0001.png ~ dashboard_0279.png
- 第二批: dashboard_0280.png ~ dashboard_0350.png (假设70张)
```

### Step 2: 放置文件

```bash
# 图片放到raw目录
data/raw/batch_002/
├── dashboard_0280.png
├── dashboard_0281.png
└── ...

# 标注放到annotations目录（按标注者分组）
data/annotations/batch_002/
├── annotator1/
│   └── 0280-0320.xml
└── annotator2/
    └── 0321-0350.xml
```

### Step 3: 运行转换脚本

```bash
# 自动转换为YOLO格式并合并到训练集
python scripts/convert_to_yolo_format.py

# 或使用新的批次转换工具（推荐）
python scripts/add_new_batch.py --batch batch_002
```

### Step 4: 验证数据

```bash
# 检查数据完整性
python scripts/validate_data.py --batch batch_002

# 查看数据统计
python scripts/show_data_stats.py
```

### Step 5: 重新训练

```bash
# 使用全部数据重新训练
START_PHASE1_IMPROVED_TRAINING.bat
```

---

## 🔧 数据索引表格式

`data/data_index.csv`:

```csv
image_id,batch_id,filename,width,height,annotator,annotation_file,split,num_objects,added_date
0001,batch_001,dashboard_0001.png,1920,1080,annotator1,0001-0069.xml,train,5,2025-09-20
0002,batch_001,dashboard_0002.png,1920,1080,annotator1,0001-0069.xml,test,3,2025-09-20
...
0280,batch_002,dashboard_0280.png,1920,1080,annotator1,0280-0320.xml,train,4,2025-10-16
```

**字段说明**:
- `image_id`: 图片ID（去掉dashboard_和.png）
- `batch_id`: 批次ID
- `filename`: 文件名
- `width/height`: 图片尺寸
- `annotator`: 标注者
- `annotation_file`: 对应的XML文件
- `split`: 数据集划分（train/val/test）
- `num_objects`: 标注对象数量
- `added_date`: 添加日期

---

## 📊 数据统计格式

`data/statistics/data_overview.json`:

```json
{
  "total_images": 350,
  "total_batches": 2,
  "batches": {
    "batch_001": {
      "images": 279,
      "objects": 1234,
      "date_added": "2025-09-20"
    },
    "batch_002": {
      "images": 71,
      "objects": 312,
      "date_added": "2025-10-16"
    }
  },
  "split_distribution": {
    "train": 206,
    "val": 62,
    "test": 62
  },
  "class_distribution": {
    "Card": 450,
    "Bar chart": 280,
    "Line chart": 47,
    "Heatmap": 15,
    ...
  }
}
```

---

## 🎯 类别优先级策略

基于当前模型性能，优先收集以下类别数据：

### 🔴 紧急需要（样本<20）
- **Pie chart** (当前1个) → 目标: 30+
- **Radar chart** (当前1个) → 目标: 30+
- **Heatmap** (当前5个) → 目标: 50+
- **Scatter plot** (当前11个) → 目标: 40+
- **Line chart** (当前17个) → 目标: 60+

### 🟡 适量增加（样本20-50）
- **Timeline** (当前1个) → 目标: 30+
- **其他稀有类别**

### 🟢 维持平衡
- **Card** (最多)
- **Bar chart** (较多)
- 其他已有足够样本的类别

---

## 🚀 自动化工具脚本

### 工具1: 批次添加脚本

`scripts/add_new_batch.py`:

```python
"""
自动化添加新批次数据

用法:
python scripts/add_new_batch.py --batch batch_002 --priority-classes "Pie chart,Heatmap"
"""
```

功能:
- ✅ 自动检测批次编号
- ✅ 验证文件命名规范
- ✅ 转换为YOLO格式
- ✅ 更新数据索引
- ✅ 生成统计报告
- ✅ 检测优先类别占比

### 工具2: 数据验证脚本

`scripts/validate_data.py`:

```python
"""
验证数据完整性和规范性

用法:
python scripts/validate_data.py --batch batch_002
"""
```

检查项:
- ✅ 图片和标注文件一一对应
- ✅ 文件命名规范性
- ✅ XML格式正确性
- ✅ 图片完整性（可读取、尺寸正常）
- ✅ 标注边界框合法性（在图片范围内）

### 工具3: 数据统计脚本

`scripts/show_data_stats.py`:

```python
"""
显示数据集统计信息

用法:
python scripts/show_data_stats.py --detailed --export report.pdf
"""
```

输出:
- 📊 总体数据量
- 📊 各类别分布
- 📊 各批次信息
- 📊 数据集划分比例
- 📊 标注质量指标

### 工具4: 数据清理脚本

`scripts/clean_duplicates.py`:

```python
"""
检测和清理重复数据

用法:
python scripts/clean_duplicates.py --similarity 0.95
"""
```

功能:
- 🔍 基于图像相似度检测重复
- 🔍 检测文件名冲突
- 🗑️ 安全移除重复项（备份到trash/）

---

## 📝 数据质量检查清单

### 添加新数据前:
- [ ] 确认批次编号不重复
- [ ] 图片命名符合规范
- [ ] 图片格式为PNG
- [ ] 图片清晰可读

### 标注质量检查:
- [ ] 标注覆盖所有可见图表
- [ ] 边界框准确（不过大/过小）
- [ ] 类别标注正确
- [ ] 没有遗漏的对象

### 转换后验证:
- [ ] 运行验证脚本无错误
- [ ] 检查YOLO标签文件
- [ ] 查看数据统计报告
- [ ] 确认优先类别有增加

---

## 🎯 推荐工作流程

### 定期数据收集（推荐：每周/每两周）

```bash
# 1. 收集新数据（50-100张为一批）
# 2. 标注数据（使用CVAT）
# 3. 下载标注并组织文件

# 4. 添加到项目
python scripts/add_new_batch.py --batch batch_XXX

# 5. 验证数据
python scripts/validate_data.py --batch batch_XXX

# 6. 查看统计（决定是否训练）
python scripts/show_data_stats.py

# 7. 如果新增数据>50张，重新训练
START_PHASE1_IMPROVED_TRAINING.bat

# 8. 评估新模型
python scripts/test_ultralytics.py

# 9. 如果性能提升，更新部署模型
```

---

## ⚠️ 注意事项

1. **不要删除raw目录的文件** - 这是唯一的原始数据源
2. **每批数据独立目录** - 方便追溯和管理
3. **保持命名连续性** - dashboard_XXXX编号不跳跃
4. **及时备份** - 定期备份raw和annotations目录
5. **记录变更** - 在`data/CHANGELOG.md`中记录每次数据更新

---

## 📈 目标数据量规划

### 短期目标（1-2个月）
- 总数据量: 350-500张
- 重点类别各30+样本

### 中期目标（3-6个月）
- 总数据量: 800-1000张
- 所有类别均衡（各50+样本）
- mAP50 目标: 65-70%

### 长期目标（6-12个月）
- 总数据量: 1500-2000张
- 覆盖各种边界情况
- mAP50 目标: 75-80%

---

**版本**: v1.0  
**创建日期**: 2025-10-16  
**维护者**: 项目团队

