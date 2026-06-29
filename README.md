# 图书馆书架图书检测项目

## 项目简介

本项目旨在开发一个基于深度学习的图书馆书架图书检测与书脊分割系统。通过采集图书馆书架视频，提取关键帧并进行标注，训练实例分割模型来自动识别和分割书脊区域。

**项目成员**: 侯舒扬、张梓晴  
**项目时间**: 2026年6月  
**GitHub仓库**: [ShuyangHou/book](https://github.com/ShuyangHou/book)

---

## 项目结构

```
项目制实践-AI/
├── dataset/              # 原始视频数据(90个MP4文件, 约5.6GB)
├── frames/               # 帧图片(229张) + JSON标注文件(229个)
├── book_spine_dataset/   # COCO格式训练数据集
│   └── coco/
│       ├── train/        # 训练集(39张)
│       └── val/          # 验证集(10张)
├── _tools/               # 数据处理工具链
├── annotations/          # 标注清单和状态跟踪
├── thumbnails/           # 视频缩略图索引
├── LABELME_标注指南.md   # 标注详细指南
└── 项目分工说明.md        # 团队分工文档
```

**说明**: 
- `frames/` 目录已整合所有图片和标注文件，包含预标注和手工标注
- 标注工作直接在 `frames/` 目录中进行，使用 LabelMe 打开修改即可

---

## 标注指令

### 1. 环境准备

#### 安装 LabelMe

```bash
# 使用 pip 安装
pip install labelme

# 或使用 conda 安装
conda install -c conda-forge labelme
```

#### 验证安装

```bash
labelme --version
```

### 2. 标注流程

#### Step 1: 启动 LabelMe

```bash
# 方式1: 直接启动LabelMe图形界面
labelme

# 方式2: 指定frames目录(推荐)
labelme frames/

# JSON文件会自动保存到frames/目录，与图片同名
```

#### Step 2: 打开图片

1. 点击菜单 `File` → `Open` 或按 `Ctrl+O`
2. 选择 `frames/` 目录下的图片文件
3. 或直接拖拽图片到 LabelMe 窗口
4. **如果该图片已有JSON标注文件，LabelMe会自动加载并显示现有标注**

#### Step 3: 创建多边形标注

**标注对象**: 书脊区域

1. **选择工具**:
   - 点击 `Edit` → `Create Polygons` 或按快捷键 `Ctrl+N`
   
2. **绘制多边形**:
   - 沿着书脊的边缘依次点击鼠标左键，创建多边形顶点
   - 尽量贴合书脊边缘，保持标注精确
   - 完成后右键点击或按 `Enter` 键闭合多边形
   
3. **填写标签**:
   - 弹出对话框时，输入标签名称: `book_spine`
   - 点击 `OK` 确认

4. **继续标注**:
   - 对图片中的每一本书的书脊重复步骤2-3
   - 一张图片通常包含多个书脊实例

#### Step 4: 编辑标注（可选）

- **移动顶点**: 选中多边形后，拖动顶点调整位置
- **删除标注**: 选中多边形，按 `Delete` 键
- **修改标签**: 右键点击标注 → `Edit Label`

#### Step 5: 保存标注

1. 点击 `File` → `Save` 或按 `Ctrl+S`
2. JSON文件会自动保存到图片同目录或指定的输出目录
3. 文件名格式: `图片名.json`（如 `046_05s.json`）

#### Step 6: 切换到下一张图片

1. 点击 `File` → `Next Image` 或按 `D` 键
2. 继续重复 Step 3-5

### 3. 标注规范

#### 标签命名
- **统一使用**: `book_spine`（小写，下划线分隔）
- **禁止使用**: `book`, `spine`, `书脊` 等其他变体

#### 标注质量要求

1. **边界精确**: 
   - 多边形边缘应紧贴书脊边界
   - 顶点数量适中（8-15个为宜）
   - 避免过度简化或过度复杂

2. **完整性**:
   - 标注图片中所有可见的书脊
   - 包括部分遮挡的书脊（至少可见50%以上）
   - 忽略完全被遮挡或模糊不清的书脊

3. **一致性**:
   - 保持标注风格一致
   - 相同情况采用相同的标注策略

#### 特殊情况处理

| 情况 | 处理方式 |
|------|----------|
| 书脊部分被遮挡 | 标注可见部分（>50%可见） |
| 书脊模糊不清 | 不标注，跳过 |
| 书脊倾斜或扭曲 | 按实际轮廓标注 |
| 多本书紧密排列 | 分别标注每本书的书脊 |
| 横放的书 | 仍然标注其书脊（侧面） |

### 4. 快捷键参考

| 功能 | 快捷键 |
|------|--------|
| 打开文件 | `Ctrl+O` |
| 保存标注 | `Ctrl+S` |
| 创建多边形 | `Ctrl+N` |
| 下一张图片 | `D` |
| 上一张图片 | `A` |
| 删除选中标注 | `Delete` |
| 撤销 | `Ctrl+Z` |
| 放大 | `Ctrl++` |
| 缩小 | `Ctrl+-` |
| 适应窗口 | `Ctrl+0` |

### 5. 分工安排

- **侯舒扬**: 负责标注视频 046-067（共90个JSON文件）
- **张梓晴**: 负责标注视频 068-090（共90个JSON文件）

### 6. 标注进度跟踪

标注完成后，检查以下内容：

```bash
# 统计已完成的标注数量
python _tools/count_labels.py

# 查看标注状态
cat annotations/label_status.csv
```

### 7. 常见问题

**Q: LabelMe 闪退或无法启动？**  
A: 检查 Python 版本（推荐 3.7-3.10），重新安装 `pip install --upgrade labelme`

**Q: JSON 文件保存位置不对？**  
A: 启动时使用 `--output` 参数指定输出目录

**Q: 如何批量处理？**  
A: 使用 `labelme 目录路径/` 打开整个文件夹，用 `D/A` 键快速切换图片

**Q: 标注错误如何修改？**  
A: 重新打开对应的图片文件，LabelMe 会自动加载已有的 JSON 标注，修改后保存即可

---

## 数据处理工具

项目提供了完整的数据处理 pipeline（位于 `_tools/` 目录）：

### 数据预处理

```bash
# 1. 视频质量扫描
python _tools/quality_scan.py

# 2. 提取关键帧
python _tools/reextract_bad_frames.py

# 3. 选取高质量帧
python _tools/pick_top50.py
```

### 标注辅助

```bash
# 4. 自动预标注（OpenCV）
python _tools/auto_label_opencv.py

# 5. 统计标注数量
python _tools/count_labels.py

# 6. 检查并修复标注
python _tools/check_and_fix_labels.py
```

### 数据集构建

```bash
# 7. 构建训练/验证数据集
python _tools/build_dataset.py

# 8. 转换为 COCO 格式
python _tools/labelme2coco.py
```

### 模型训练

```bash
# 9. 安装训练依赖
pip install -r _tools/requirements_train.txt

# 10. 训练 MaskDINO 模型
python _tools/train_maskdino_r50.py

# 11. 批量预测剩余帧
python _tools/predict_remaining.py
```

**详细流程**: 参见 [`_tools/PIPELINE.md`](_tools/PIPELINE.md)

---

## 数据集信息

### 原始数据
- **视频数量**: 90个 MP4 文件
- **视频总大小**: 约 5.6GB
- **视频来源**: 图书馆书架实地拍摄

### 处理后数据
- **提取帧**: 180张关键帧（1920×1080）
- **精选帧**: 50张高质量帧用于重点标注
- **预标注**: 180个 LabelMe 格式 JSON 文件

### 训练数据集
- **格式**: COCO 实例分割格式
- **训练集**: 39张图片 + `instances_train.json`
- **验证集**: 10张图片 + `instances_val.json`
- **类别**: 1个（book_spine）

---

## Git 大文件管理

项目使用 Git LFS 管理大文件（视频数据集）。

### 克隆仓库

```bash
# 安装 Git LFS
git lfs install

# 克隆仓库
git clone https://github.com/ShuyangHou/book.git
cd book

# 拉取 LFS 文件
git lfs pull
```

### 查看 LFS 文件

```bash
git lfs ls-files
```

---

## 技术栈

- **标注工具**: LabelMe
- **深度学习框架**: PyTorch
- **分割模型**: MaskDINO (ResNet-50 backbone)
- **数据格式**: COCO JSON
- **版本控制**: Git + Git LFS
- **开发语言**: Python 3.8+

---

## 参考文档

- [LabelMe 官方文档](https://github.com/wkentaro/labelme)
- [COCO 数据格式说明](https://cocodataset.org/#format-data)
- [MaskDINO 论文](https://arxiv.org/abs/2206.02777)
- [项目详细标注指南](LABELME_标注指南.md)
- [项目分工说明](项目分工说明.md)

---

## 常用命令速查

```bash
# 启动标注
labelme frames_pick50/ --output prelable/

# 统计标注
python _tools/count_labels.py

# 构建数据集
python _tools/build_dataset.py
python _tools/labelme2coco.py

# 训练模型
python _tools/train_maskdino_r50.py

# 查看标注状态
cat annotations/label_status.csv
```

---

## 许可证

本项目仅用于学术研究和课程实践。

---

**最后更新**: 2026年6月29日




