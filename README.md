# 图书盘点项目

本项目用于课程《项目制实践》的图书盘点方案整理，当前本地目录只保留仍在使用的代码、数据、标注和说明文档。模型大产物已经转移到服务器，本地以流程脚本、真值表和方案文档为主。

## 当前目录

```text
项目制实践-AI/
├── dataset/              # 原始视频与馆藏 Excel
├── frames/               # 关键帧与对应 LabelMe JSON
├── book_spine_dataset/   # COCO 数据集
├── annotations/
│   └── video_gt.csv      # 视频级真值表
├── _tools/
│   ├── check_and_fix_labels.py
│   ├── count_labels.py
│   ├── labelme2coco.py
│   ├── train_maskdino_r50.py
│   ├── ocr_qianfan.py
│   ├── build_pred_inventory.py
│   ├── score_video_inventory.py
│   └── requirements_train.txt
└── docs/
    ├── 2026春季学期项目制实践实验指导书.md
    ├── 整体方案.md
    └── 机器初稿生成说明.md
```

## 主要脚本

- `python _tools/count_labels.py`
  统计当前标注数量。
- `python _tools/check_and_fix_labels.py --dir frames --fix`
  检查并修复 LabelMe 标注格式问题。
- `python _tools/labelme2coco.py`
  把 LabelMe 标注转换成 COCO 数据集。
- `python _tools/train_maskdino_r50.py`
  训练书脊分割模型。
- `python _tools/ocr_qianfan.py`
  调用百度千帆 `deepseek-ocr` 识别书脊文本。
- `python _tools/build_pred_inventory.py --ocr-csv <你的OCR结果.csv>`
  汇总 OCR 结果，生成机器初稿 `pred_inventory.csv`。
- `python _tools/score_video_inventory.py --pred <pred_inventory.csv>`
  用 `annotations/video_gt.csv` 评估视频级统计结果。

## OCR 鉴权

脚本不内置密钥，运行前自行设置：

```powershell
$env:QIANFAN_API_KEY="YOUR_API_KEY"
```

## 说明

- `ocr_qianfan.py` 和 `build_pred_inventory.py` 现在都支持“不带馆藏词典”直接运行。
- 本地目录已经删除大部分中间产物、历史脚本、缩略图、模板附件和权重文件。
- 如果后面还需要恢复候选词典，可以再从 `dataset/` 里的馆藏 Excel 重新生成。
