import re
from PIL import Image
import numpy as np

from common.preprocess import transform

image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

def get_image_files(img_path):
    """返回 input_imgs 文件夹中所有支持的图片文件路径列表【Path对象】"""
    return [f for f in img_path.iterdir() if f.suffix.lower() in image_extensions]

def extract_true_label(filename: str) -> int:
    """从文件名中提取 [数字] 作为真实标签，例如 img_40[2].png -> 2"""
    match = re.search(r'\[(\d+)\]', filename)
    if not match:
        raise ValueError(f"文件名 {filename} 中未找到形如 [数字] 的正确标签")
    return int(match.group(1))

def load_test_img(image_paths):
    """批量加载测试图片【返回形式：张量列表】"""
    images = []
    for p in image_paths:
        img = Image.open(p)
        img_tensor = transform(img).unsqueeze(0) #单个张量
        images.append(img_tensor.numpy())
    batch_image = np.concatenate(images, axis=0).astype(np.float32) #batch个张量合并
    return batch_image
