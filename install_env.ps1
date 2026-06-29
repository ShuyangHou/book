# 安装 PyTorch 和 detectron2 的脚本
# 使用方法: 
# 1. conda activate book-spine
# 2. 运行此脚本中的命令

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install opencv-python pillow numpy

# 安装 detectron2 (从源码,因为没有预编译的 Windows wheel)
pip install git+https://github.com/facebookresearch/detectron2.git

# 或者如果上面失败,尝试:
# python -m pip install "git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2"
