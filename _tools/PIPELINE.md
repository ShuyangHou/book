# 书脊分割 Pipeline 使用指南

## 整体流程

```
50张精标 Labelme
→ 标注检查 (check_and_fix_labels.py)
→ 转 COCO 格式 (labelme2coco.py)
→ 训练 MaskDINO-R50 第一版 (train_maskdino_r50.py)
→ 预测剩余帧 (predict_remaining.py)
→ 人工修正预标注
→ 扩到 160~220 张
→ 训练最终 MaskDINO-SwinL
→ 接 DeepSeek-OCR
```

## 第一步：检查并修复标注

```bash
# 仅检查（不修改文件）
python _tools/check_and_fix_labels.py --dir frames_pick50

# 检查并自动修复
python _tools/check_and_fix_labels.py --dir frames_pick50 --fix
```

自动修复内容：
- `rectangle` / `oriented_rectangle` → `polygon`
- 2点退化 polygon → 删除
- 自交叉 polygon → `shapely.make_valid` 修复

## 第二步：Labelme 转 COCO

```bash
python _tools/labelme2coco.py \
    --src frames_pick50 \
    --dst book_spine_dataset/coco \
    --val-ratio 0.2 \
    --seed 42
```

输出结构：
```
book_spine_dataset/coco/
  train/
    images/        (39 张)
    instances_train.json
  val/
    images/        (10 张)
    instances_val.json
```

类别：`{"id": 1, "name": "book_spine"}`

按视频编号划分 train/val，避免同一视频的相似帧泄露。

## 第三步：训练 MaskDINO-R50 第一版

### 环境准备（需要 GPU）

```bash
# PyTorch (根据你的 CUDA 版本选择)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html

# MaskDINO
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO && pip install -e .

# 其他依赖
pip install -r _tools/requirements_train.txt
```

### 开始训练

```bash
python _tools/train_maskdino_r50.py \
    --data-root book_spine_dataset/coco \
    --output-dir output/maskdino_r50_v1 \
    --num-gpus 1 \
    --max-iter 3000 \
    --batch-size 2 \
    --lr 0.0001
```

如果 MaskDINO 安装有问题，脚本会自动回退到 **Mask R-CNN R50-FPN**（同样有效）。

训练完成后模型在 `output/maskdino_r50_v1/model_final.pth`。

## 第四步：预测剩余帧

```bash
python _tools/predict_remaining.py \
    --model-dir output/maskdino_r50_v1 \
    --frames-dir frames \
    --already-labeled frames_pick50 \
    --output-dir frames_remaining_prelabel \
    --threshold 0.5
```

输出 Labelme 格式的 JSON，可直接在 Labelme 里打开修正。

### 人工修正流程

1. 用 Labelme 打开 `frames_remaining_prelabel/` 目录
2. 逐张检查：删错的、补漏的、调边界
3. 保存修正后的标注
4. 合并到训练集重新转 COCO

## 第五步：扩充后最终训练

扩到 160~220 张后：

```bash
# 重新转 COCO（调整划分）
python _tools/labelme2coco.py \
    --src frames_all_labeled \
    --dst book_spine_dataset_final/coco \
    --val-ratio 0.15

# 训练最终 MaskDINO-SwinL（需要修改 train 脚本的 config）
```

## 当前数据统计

| 项目 | 数量 |
|------|------|
| 精标图片（有 polygon） | 49 |
| 空标注 JSON | 25 |
| 缺失 JSON | 1 (038_17s.jpg) |
| 总 polygon 数 | 1279 |
| Train | 39 (1004 anns) |
| Val | 10 (275 anns) |
| 类别 | book_spine |
