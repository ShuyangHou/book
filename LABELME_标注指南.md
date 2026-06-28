# Labelme 标注指南

## 启动 Labelme

```powershell
cd C:\Users\15075\Desktop\课程作业\项目制实践-AI
python -m labelme frames_pick50 --labels book_spine --autosave
```

## 标注流程（推荐用 AI Polygon）

### 方法一：AI 辅助标注（推荐）

1. 打开 Labelme 后，点左侧工具栏 **AI Polygon** 或按快捷键
2. **点击书脊中心一次**，EfficientSAM 会自动生成轮廓
3. 检查轮廓是否准确：
   - 如果准确：直接保存（Ctrl+S）
   - 如果不准：手动调整多边形顶点，或删除重标
4. 标签统一选 `book_spine`
5. 标完一张按 `D` 跳到下一张

### 方法二：纯手动标注

1. 点左侧 **Create Polygon** 或按 `Ctrl+N`
2. 沿书脊边缘依次点击顶点（顺时针或逆时针）
3. 标完一本书后右键或双击闭合
4. 选择标签 `book_spine`
5. 继续标下一本书

## 标注要求

- **每本书的书脊标一个多边形**
- **紧贴书脊边缘**，不要把书架缝隙/其他书的部分框进来
- **倾斜的书也要标**
- **部分遮挡的书可以标可见部分**
- **书名看不清没关系，只要书脊轮廓清楚就标**
- 如果整张图书脊都看不清（比如全是走廊/墙壁），可以跳过不标

## 保存格式

Labelme 会为每张图片生成一个同名 `.json` 文件：

```
frames_pick50/
  001_03s.jpg       ← 原图
  001_03s.json      ← 标注文件（自动保存）
  002_05s.jpg
  002_05s.json
  ...
```

## 标注进度跟踪

标完一部分后可以跑这个统计：

```powershell
python -c "import os; jsons = [f for f in os.listdir('frames_pick50') if f.endswith('.json')]; print(f'已标注: {len(jsons)}/50')"
```

## 预计标注时间

- 用 AI Polygon：每张 1-2 分钟，50 张约 1-2 小时
- 纯手动：每张 3-5 分钟，50 张约 3-4 小时

## 标完之后

标完 50 张后可以：

1. 训练 YOLOv8-seg 第一版模型
2. 用模型给剩下 112 张关键帧预标注
3. 人工修正预标注结果
4. 继续标注 + 训练迭代

---

**Tips**:
- 标注时放大图片看得更清楚（滚轮缩放）
- 每标 10 张可以休息一下，避免眼疲劳
- 不要追求完美，80% 准确度就够训练第一版模型
