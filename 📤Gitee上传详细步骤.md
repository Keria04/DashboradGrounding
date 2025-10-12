# 📤 Gitee（码云）上传详细步骤

## 🎯 优势

- ✅ **国内平台，访问快速稳定**
- ✅ **免费无限私有仓库**
- ✅ **可以自动同步到GitHub**
- ✅ **操作简单，中文界面**
- ✅ **支持Git LFS（免费5GB）**

---

## 🚀 完整步骤

### 步骤1：注册Gitee账号（3分钟）

1. 访问：**https://gitee.com/**
2. 点击右上角 "**注册**"
3. 填写信息：
   ```
   手机号/邮箱
   用户名（会成为仓库地址的一部分）
   密码
   ```
4. 验证并完成注册
5. 登录成功 ✅

---

### 步骤2：创建仓库（2分钟）

1. 登录后，点击右上角 "**+**" → "**新建仓库**"
2. 填写信息：
   ```
   仓库名称: DashboardDetection
   （或 Project_srp，随您喜欢）
   
   路径: dashboard-detection
   
   仓库介绍: AI仪表盘图表检测系统 - YOLOv8
   
   是否开源: 
   ○ 公开（推荐，团队成员可访问）
   ○ 私有（只有您可见）
   
   初始化:
   ☐ 不要勾选 "使用Readme文件初始化仓库"
   ☐ 不要选择 .gitignore 和 License
   ```
3. 点击 "**创建**"

**获得仓库地址**：
```
https://gitee.com/你的用户名/dashboard-detection.git
```

---

### 步骤3：推送代码到Gitee（5分钟）

在命令行（PowerShell）中运行：

```bash
# 1. 移除旧的GitHub remote
git remote remove origin

# 2. 添加Gitee仓库
git remote add origin https://gitee.com/你的用户名/dashboard-detection.git

# 3. 推送
git push -u origin main
```

**输入凭据**：
```
Username: 你的Gitee用户名
Password: 你的Gitee登录密码（不需要Token，直接用密码！）
```

**等待上传**：5-10分钟（主要是模型文件21.5MB）

---

### 步骤4：配置自动同步到GitHub（可选，5分钟）

**让Gitee自动将代码同步到GitHub团队仓库**：

1. 在Gitee仓库页面，点击 "**管理**"
2. 左侧菜单 → "**仓库镜像管理**"
3. 点击 "**添加镜像**"
4. 填写：
   ```
   镜像方向: Push（推送）
   
   远程仓库地址: 
   https://github.com/Keria04/DashboradGrounding.git
   
   用户名: 你的GitHub用户名
   
   密码/令牌: 你的GitHub Personal Access Token
   （如果没有，暂时跳过此步骤）
   ```
5. 点击 "**添加**"

**配置成功后**：
- 每次推送到Gitee
- 自动同步到GitHub
- 团队成员在GitHub看到更新

---

## ✅ 上传成功后

### 验证上传

访问：
```
https://gitee.com/你的用户名/dashboard-detection
```

应该看到：
- ✅ README.md显示项目说明
- ✅ scripts/文件夹
- ✅ experiments/文件夹
- ✅ 模型文件（LFS标记）

### 分享给团队

**Gitee链接**：
```
https://gitee.com/你的用户名/dashboard-detection
```

**如果配置了GitHub同步**：
- 团队成员在GitHub也能看到：
  ```
  https://github.com/Keria04/DashboradGrounding
  ```

---

## 🔧 Gitee特有功能

### 1. Pages服务（免费）

可以部署静态网站展示项目：
```
服务 → Gitee Pages → 部署/更新
```

### 2. WebIDE

在线编辑代码：
```
点击任意文件 → WebIDE
```

### 3. 与GitHub双向同步

既可以从GitHub导入，也可以推送到GitHub

---

## 📊 Gitee vs GitHub 对比

| 特性 | Gitee | GitHub |
|------|-------|--------|
| **国内访问** | ✅ 快 | ❌ 慢/不稳定 |
| **免费私有仓库** | ✅ 无限 | ✅ 无限 |
| **LFS免费额度** | 5GB | 1GB |
| **Pages服务** | ✅ 免费 | ✅ 免费 |
| **同步GitHub** | ✅ 支持 | - |
| **国际影响力** | 一般 | 高 |

---

## 🎯 推荐工作流程

### 日常开发（使用Gitee）

```bash
# 修改代码
git add .
git commit -m "更新说明"
git push

# 推送到Gitee（快速）
# 自动同步到GitHub（如果配置了）
```

### 团队协作

```
您 → Gitee（主要工作）
      ↓ 自动同步
   GitHub（团队查看）
```

---

## ⚡ 立即开始

### 现在就可以操作：

```bash
# 1. 访问 https://gitee.com/ 注册

# 2. 创建仓库

# 3. 在命令行运行：
git remote remove origin
git remote add origin https://gitee.com/你的用户名/dashboard-detection.git
git push -u origin main

# 4. 输入Gitee用户名和密码
```

**5-10分钟后上传完成！** ✅

---

## 📞 常见问题

### Q: Gitee需要Token吗？

A: **不需要！** 直接用登录密码即可
```
Username: 你的Gitee用户名
Password: 你的Gitee登录密码
```

### Q: 模型文件太大怎么办？

A: Gitee免费支持5GB LFS，完全够用
```bash
git lfs install
git lfs track "*.pt"
git push
```

### Q: 如何同步到GitHub？

A: Gitee仓库 → 管理 → 仓库镜像管理 → 添加GitHub同步

### Q: 团队成员如何访问？

A: 
- Gitee: 分享 `https://gitee.com/你的用户名/dashboard-detection`
- GitHub: 配置同步后，他们在GitHub也能看到

---

## 🎊 总结

**Gitee是最适合您的方案**：
- 🌐 国内访问快
- 🔓 简单（用密码不用Token）
- 🔄 可以同步到GitHub
- 💰 完全免费

---

**现在就访问 https://gitee.com/ 注册吧！** 🚀

注册完成后告诉我，我帮您完成后续配置！

