"""
重新提取缺失的帧图片
从frames_pick50的JSON文件列表中提取对应的视频帧
"""

import cv2
import os
from pathlib import Path

def extract_frame(video_path, timestamp_seconds, output_path):
    """从视频中提取指定时间戳的帧"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    success, frame = cap.read()
    cap.release()
    if success:
        cv2.imwrite(str(output_path), frame)
        return True
    return False

def main():
    frames_pick50_dir = Path("frames_pick50")
    dataset_dir = Path("dataset")
    json_files = []
    for json_file in frames_pick50_dir.glob("*.json"):
        video_num = int(json_file.stem.split('_')[0])
        if 1 <= video_num <= 45:
            json_files.append(json_file)
    print(f"找到 {len(json_files)} 个需要提取帧的JSON文件")
    success_count = 0
    fail_count = 0
    skip_count = 0
    for json_file in sorted(json_files):
        stem = json_file.stem
        parts = stem.split('_')
        video_num = parts[0]
        time_str = parts[1]
        seconds = int(time_str.replace('s', ''))
        video_path = dataset_dir / f"{video_num}.mp4"
        output_path = frames_pick50_dir / f"{stem}.jpg"
        if output_path.exists():
            skip_count += 1
            continue
        if not video_path.exists():
            print(f"视频不存在: {video_path}")
            fail_count += 1
            continue
        if extract_frame(video_path, seconds, output_path):
            print(f"提取成功: {stem}.jpg")
            success_count += 1
        else:
            print(f"提取失败: {stem}.jpg")
            fail_count += 1
    print("\n" + "="*50)
    print(f"提取完成: 成功{success_count} 失败{fail_count} 跳过{skip_count} 总计{len(json_files)}")

if __name__ == "__main__":
    main()
