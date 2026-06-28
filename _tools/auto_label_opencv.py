"""
使用 OpenCV 边缘检测自动标注书脊（本地运行，不依赖 PyTorch）
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

FRAMES_DIR = Path("frames_pick50")

def detect_book_spines(image_path):
    """使用 OpenCV 检测竖条状书脊"""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 形态学闭运算，连接断裂的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    book_spines = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # 书脊特征筛选
        aspect_ratio = ch / (cw + 1e-6)
        width_ratio = cw / w
        height_ratio = ch / h
        
        # 竖条：高宽比 > 2.5，宽度占 2-20%，高度占 20% 以上
        if (aspect_ratio > 2.5 and 
            0.02 < width_ratio < 0.2 and 
            height_ratio > 0.2):
            
            # 转为多边形（简化轮廓）
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 如果顶点太多，用外接矩形
            if len(approx) > 8:
                points = [[x, y], [x+cw, y], [x+cw, y+ch], [x, y+ch]]
            else:
                points = approx.reshape(-1, 2).tolist()
            
            book_spines.append({
                "points": points,
                "score": aspect_ratio
            })
    
    # 按置信度排序，取前 20 个
    book_spines.sort(key=lambda x: x["score"], reverse=True)
    return book_spines[:20]

def create_labelme_json(image_path, detections):
    """生成 Labelme JSON"""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    shapes = []
    for det in detections:
        shapes.append({
            "label": "book_spine",
            "points": det["points"],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    
    return {
        "version": "6.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

def auto_label_all():
    """自动标注所有图片"""
    images = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"找到 {len(images)} 张图片\n")
    
    success = 0
    skipped = 0
    
    for img_path in tqdm(images, desc="自动标注"):
        json_path = img_path.with_suffix(".json")
        
        if json_path.exists():
            skipped += 1
            continue
        
        try:
            detections = detect_book_spines(img_path)
            labelme_data = create_labelme_json(img_path, detections)
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            
            success += 1
        except Exception as e:
            print(f"\n✗ {img_path.name}: {e}")
    
    print(f"\n✓ 完成: {success} 张新标注, {skipped} 张已存在")
    return success

if __name__ == "__main__":
    print("=" * 60)
    print("OpenCV 自动标注书脊")
    print("=" * 60)
    print("\n注意：边缘检测准确率 40-60%，需要手动修正\n")
    
    input("按回车开始标注...")
    
    success = auto_label_all()
    
    if success > 0:
        print("\n" + "=" * 60)
        print("接下来：")
        print("1. 打开 Labelme 检查标注：")
        print("   python -m labelme frames_pick50 --labels book_spine --autosave")
        print("\n2. 手动修正不准确的标注")
        print("3. 精标 10-20 张后可以训练 YOLOv8-seg")
        print("=" * 60)
