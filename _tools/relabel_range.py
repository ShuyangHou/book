# -*- coding: utf-8 -*-
"""对指定范围的图片用训练好的模型做预标注，覆盖现有JSON"""
import argparse
import json
import os
import base64
from pathlib import Path
import cv2
import numpy as np


def load_predictor(model_path: str, threshold: float):
    """加载 Detectron2 预测器"""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()
    
    # 尝试加载 MaskDINO
    try:
        from maskdino import add_maskdino_config
        add_maskdino_config(cfg)
    except ImportError:
        pass

    # 使用 Mask R-CNN 配置
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda"  # 如果没 GPU 改成 "cpu"
    
    return DefaultPredictor(cfg)


def mask_to_polygon(mask):
    """将 mask 转为 polygon 坐标"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    
    # 取最大轮廓
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None
    
    # 简化轮廓
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
    return polygon


def predict_to_labelme(predictor, img_path: str, threshold: float):
    """预测并生成 Labelme JSON"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    outputs = predictor(img)
    
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    
    shapes = []
    for i in range(len(masks)):
        if scores[i] < threshold:
            continue
        
        polygon = mask_to_polygon(masks[i])
        if polygon is None or len(polygon) < 3:
            continue
        
        shapes.append({
            "label": "book_spine",
            "points": polygon,
            "group_id": None,
            "description": f"auto_conf={scores[i]:.3f}",
            "shape_type": "polygon",
            "flags": {},
        })
    
    # 编码图片为 base64
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": img_data,
        "imageHeight": h,
        "imageWidth": w,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--frames-dir", default="frames", help="图片目录")
    parser.add_argument("--start", required=True, help="起始文件名(如 027_17s)")
    parser.add_argument("--end", required=True, help="结束文件名(如 045_12s)")
    parser.add_argument("--threshold", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--no-imagedata", action="store_true", help="不嵌入图片数据")
    args = parser.parse_args()
    
    # 找到范围内的图片
    all_imgs = sorted(Path(args.frames_dir).glob("*.jpg"))
    target_imgs = [
        img for img in all_imgs 
        if args.start <= img.stem <= args.end
    ]
    
    print(f"找到 {len(target_imgs)} 张图片需要预标注")
    print(f"范围: {args.start} 到 {args.end}")
    print(f"置信度阈值: {args.threshold}")
    
    if not target_imgs:
        print("没有找到符合范围的图片！")
        return
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    predictor = load_predictor(args.model, args.threshold)
    
    # 预测
    total_spines = 0
    for i, img_path in enumerate(target_imgs):
        result = predict_to_labelme(predictor, str(img_path), args.threshold)
        if result is None:
            print(f"  [{i+1}/{len(target_imgs)}] 跳过 {img_path.name} (读取失败)")
            continue
        
        if args.no_imagedata:
            result["imageData"] = None
        
        n = len(result["shapes"])
        total_spines += n
        
        # 保存 JSON (覆盖原有的)
        json_path = img_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"  [{i+1}/{len(target_imgs)}] {img_path.name}: {n} 个书脊")
    
    print(f"\n完成! 共处理 {len(target_imgs)} 张图片, 预测 {total_spines} 个书脊")
    print(f"JSON 文件已保存到 {args.frames_dir}/")
    print("\n下一步: 用 LabelMe 打开 frames/ 检查和修正预标注结果")


if __name__ == "__main__":
    main()
