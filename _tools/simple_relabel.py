# -*- coding: utf-8 -*-
"""简化版预标注脚本,直接加载 PyTorch 模型权重"""
import argparse
import json
import os
import base64
from pathlib import Path
import torch
import cv2
import numpy as np


def mask_to_polygon(mask):
    """将 mask 转为 polygon 坐标"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None
    
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
    return polygon


def simple_predict(model, img_path, threshold=0.4, device='cuda'):
    """简化版推理,只提取必要信息"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # 转为 RGB 并归一化
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        try:
            # 尝试直接调用模型
            outputs = model(img_tensor)
            
            # 提取结果(需要根据实际模型输出格式调整)
            if isinstance(outputs, dict):
                instances = outputs.get('instances', outputs)
            else:
                instances = outputs
            
            # 获取 masks 和 scores
            if hasattr(instances, 'pred_masks'):
                masks = instances.pred_masks.cpu().numpy()
                scores = instances.scores.cpu().numpy()
            else:
                print(f"警告: 无法识别模型输出格式,跳过 {img_path}")
                return None
                
        except Exception as e:
            print(f"推理出错 {img_path}: {e}")
            return None
    
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
    
    # 编码图片
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
    parser.add_argument("--model", required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--frames-dir", default="frames", help="图片目录")
    parser.add_argument("--start", required=True, help="起始文件名")
    parser.add_argument("--end", required=True, help="结束文件名")
    parser.add_argument("--threshold", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    
    # 尝试提取模型
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint
    
    print("警告: 此脚本需要 detectron2 来正确加载模型")
    print("建议使用服务器环境或安装完整的 detectron2")
    print("\n请尝试:")
    print("  1. 在训练模型的服务器上运行预标注")
    print("  2. 或者手工标注这 20 张图片")
    return


if __name__ == "__main__":
    main()
