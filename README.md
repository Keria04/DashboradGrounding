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

