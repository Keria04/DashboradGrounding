# 🇨🇳 ModelScope创空间部署指南

**ModelScope** = 中国版的Hugging Face（阿里巴巴达摩院出品）

## 🎯 为什么选择ModelScope？

- ✅ **国内可以访问**（比Hugging Face快得多）
- ✅ **永久免费**
- ✅ **中文界面**（操作更友好）
- ✅ **类似Hugging Face**（几乎一样的操作）
- ✅ **稳定可靠**（阿里云基础设施）

---

## 📋 准备工作

### 已准备好的文件（在huggingface文件夹）

```
huggingface/
├── app.py              ✅ (Web界面代码)
├── requirements.txt    ✅ (依赖列表)
├── README.md          ✅ (项目说明)
├── .gitattributes     ✅ (Git LFS配置)
└── best.pt            ✅ (模型文件，21.5MB)
```

**注意**：ModelScope和Hugging Face使用相同的文件格式！

---

## 🚀 详细部署步骤

### 步骤1：注册ModelScope账号（3分钟）

1. 访问：https://modelscope.cn/
2. 点击右上角 "**登录/注册**"
3. 选择注册方式：
   - 手机号注册（推荐，快速）
   - 邮箱注册
   - 微信扫码登录
4. 填写信息并验证
5. 登录成功 ✅

---

### 步骤2：创建创空间（2分钟）

1. 登录后，点击右上角头像
2. 选择 "**我的**" → "**创空间**"
3. 点击 "**新建创空间**"
4. 填写信息：
   ```
   创空间名称: dashboard-chart-detector
   SDK类型: Gradio
   可见性: 公开
   硬件环境: CPU（免费）
   ```
5. 点击 "**创建**"

**创建成功后的链接**：
```
https://modelscope.cn/studios/你的用户名/dashboard-chart-detector
```

---

### 步骤3：上传文件（5分钟）

#### 方法A：网页上传（推荐）

1. 在创空间页面，点击 "**文件**" 标签
2. 点击 "**上传文件**"
3. 依次上传（从本地`huggingface/`文件夹）：
   ```
   ✅ app.py
   ✅ requirements.txt
   ✅ README.md
   ✅ .gitattributes
   ✅ best.pt (21.5MB，需要2-3分钟上传)
   ```
4. 在提交信息中输入：`初始提交`
5. 点击 "**提交**"

#### 方法B：Git命令行上传

```bash
# 1. 克隆Space仓库
git clone https://www.modelscope.cn/studios/你的用户名/dashboard-chart-detector.git
cd dashboard-chart-detector

# 2. 复制文件
cd E:\python.code\Project_srp
copy huggingface\* dashboard-chart-detector\

# 3. 提交
cd dashboard-chart-detector
git add .
git commit -m "初始提交"
git push
```

**如果best.pt太大无法推送**：
```bash
# 使用Git LFS
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add best.pt
git commit -m "添加模型"
git push
```

---

### 步骤4：等待部署（3-5分钟）

上传完成后：

1. 点击 "**应用**" 标签
2. 会看到 "**构建中...**" 状态
3. ModelScope会自动：
   - 安装Python依赖
   - 加载模型
   - 启动Gradio应用

**成功标志**：
- 状态显示 "**运行中**"（绿色）
- 可以看到您的Web界面

---

### 步骤5：测试和分享（1分钟）

1. **测试**：
   - 在应用页面上传测试图片
   - 点击"开始检测"
   - 验证功能正常

2. **分享链接**：
   ```
   https://modelscope.cn/studios/你的用户名/dashboard-chart-detector
   ```
   
   这就是您的永久公开链接！✨

---

## 📊 部署后的效果

### 您的公开网站会是这样

```
网址: https://modelscope.cn/studios/你的用户名/dashboard-chart-detector

┌─────────────────────────────────────────────────────┐
│ 🎯 仪表盘图表检测系统                                │
│                                                     │
│ AI自动识别仪表盘中的各种图表类型并精确定位          │
│                                                     │
│ 📊 模型性能: mAP50 = 51.2%, Recall = 55.7%         │
│ 🎯 支持类型: 20种图表                               │
├─────────────────────────────────────────────────────┤
│ 📤 上传图像      │  📊 检测结果                     │
│ [拖拽区域]       │  [标注图像]                      │
│ 置信度: 0.10     │  检测到3个图表:                  │
│ [开始检测]       │  1. Bar chart: 2个               │
└─────────────────────────────────────────────────────┘
```

### 访问体验

- ✅ **国内访问速度快**（阿里云CDN加速）
- ✅ **手机也可以访问**
- ✅ **无需登录**（公开访问）
- ✅ **永久有效**（不会过期）

---

## 🔧 常见问题

### Q1: 上传best.pt时提示文件太大？

**方法1**：分多次上传
- 刷新页面重试
- 或使用Git LFS

**方法2**：使用Git LFS
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes best.pt
git commit -m "添加模型"
git push
```

### Q2: 部署失败，显示错误？

点击 "**日志**" 查看错误信息

**常见错误**：

**错误1**: `No module named 'gradio'`
- 检查requirements.txt是否正确上传

**错误2**: `FileNotFoundError: best.pt`
- 确保best.pt已成功上传
- 检查文件名是否正确（大小写）

**错误3**: 构建超时
- 刷新页面重试
- 或等待几分钟

### Q3: 应用运行慢？

**原因**：免费CPU版本，推理需要5-10秒/张

**解决**：
- 接受免费版速度（够用）
- 或升级到GPU版本（付费）

### Q4: 48小时后自动休眠？

**现象**：访问时显示"启动中"

**解决**：
- 等待10-20秒自动唤醒
- 第一次访问会慢，之后就快了

---

## 🎯 ModelScope vs Gradio Share 对比

| 特性 | Gradio Share | ModelScope |
|------|--------------|------------|
| **部署时间** | 1分钟 | 15分钟 |
| **链接有效期** | 72小时 | **永久** ✅ |
| **访问速度** | 快 | **很快** ✅ |
| **需要保持运行** | 是 ⚠️ | **否** ✅ |
| **适合场景** | 快速测试 | **长期使用** ✅ |

### 推荐使用策略

```
今天: 
   用Gradio Share快速测试
   → START_WEB_APP.bat → 分享链接

本周:
   如果效果好，部署到ModelScope
   → 获得永久链接
   → 无需电脑一直运行
```

---

## 🎊 其他国内平台选择

### 1. 百度飞桨AI Studio
```
网址: https://aistudio.baidu.com/
特点: 国内平台，免费GPU
难度: ⭐⭐⭐ (需要适配PaddlePaddle格式)
```

### 2. 腾讯云开发者实验室
```
网址: https://cloud.tencent.com/developer/labs
特点: 腾讯云服务
难度: ⭐⭐⭐⭐ (需要云函数配置)
```

### 3. 阿里云函数计算
```
网址: https://www.aliyun.com/product/fc
特点: 阿里云服务，按量付费
难度: ⭐⭐⭐⭐ (配置复杂)
```

**推荐**：对于您的项目，**ModelScope是最佳选择**！

---

## 📌 快速决策

### 如果只需要测试几天：
```
→ Gradio Share（1分钟搞定）
→ START_WEB_APP.bat
→ 复制链接
```

### 如果需要长期使用：
```
→ ModelScope（15分钟搞定）
→ 注册 → 创建Space → 上传文件
→ 获得永久链接
```

---

## 🎯 立即开始

### 方案1：Gradio Share（现在就可以）

```bash
# 已经帮您启用了share=True
# 直接运行:
START_WEB_APP.bat

# 等待显示 public URL
# 复制链接发给伙伴 ✅
```

### 方案2：ModelScope（建议本周完成）

```
1. 访问 https://modelscope.cn/ 注册
2. 创建创空间
3. 上传 huggingface/ 中的5个文件
4. 获得永久链接
```

---

**我的建议**：
1. **现在**：用Gradio Share快速测试（1分钟）
2. **本周**：如果效果好，部署到ModelScope（15分钟）

**需要我提供ModelScope的详细截图教程吗？** 📸

---

**现在就运行 `START_WEB_APP.bat` 试试吧！** 🚀

