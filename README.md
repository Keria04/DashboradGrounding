# DashboradGrounding

### 目录结构

```
dashboard-visual-grounding/
├── README.md               # 项目说明文档
├── requirements.txt        # Python依赖
├── .gitignore
├── data/                   # 数据集（不直接上传，存放说明文件）
│   ├── raw/                # 原始数据（标注前）
│ 	│ 	├── 第一阶段-数据1			# 用于训练分割仪表盘的数据
│   └── annotations/        # 标注数据
│ 			├── annotatorX			# 每个负责标注的人一个文件夹，存放自己标注的数据，图片名称与raw中对应
├── docs/                   # 文档资料
├── src/                    # 核心代码
```

## 第一阶段

### Dashborad Segmentation Label

- Card （卡片/数字卡片）
- Bar chart（柱状图）
- Line chart（折线图）
- Area chart（面积图）
- Pie chart（饼图）
- Donut chart（环形图）
- Scatter plot（散点图）
- Bubble chart（气泡图）
- Rader chart（雷达图）
- Heatmap（热力图）
- Gantt chart（甘特图）
- Waterfall chart（瀑布图）
- Progress bar（进度条）
- Dial chart（计量仪表盘/速度表）
- Percentage indicator（百分比指标）
- Data table（数据表格）
- Map（地图）

#### Annotation分工

- Annotator1——张悦菲
  - dashboard_0001.png——dashboard_0069.png
- Annotator2——李昕凌
  - dashboard_0070.png——dashboard_0139.png
- Annotator3——蔡怡乐
  - dashboard_0140.png——dashboard_0209.png
- Annotator4——曹均杰
  - dashboard_0210.png——dashboard_0279.png
