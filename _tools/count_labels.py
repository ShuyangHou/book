"""统计 frames_pick50 中的标注进度"""
import json, os, glob

files = sorted(glob.glob('frames_pick50/*.json'))
total_shapes = 0
done = 0
empty = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    shapes = data.get('shapes', [])
    n = len(shapes)
    total_shapes += n
    if n > 0:
        done += 1
    else:
        empty += 1
    name = os.path.basename(f)
    print(f'{name}: {n:3d} 个标注')

print(f'\n--- 汇总 ---')
print(f'已标注帧: {done}')
print(f'未标注帧: {empty}')
print(f'总标注框: {total_shapes}')
print(f'总帧数: {len(files)}')
