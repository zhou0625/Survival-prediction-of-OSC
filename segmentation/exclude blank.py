import os
import cv2
import numpy as np


def is_empty_slice(image_path, threshold=0.99):
    """
    判断一个切片图像块是否是空的（无组织内容）。
    如果图像中的白色像素（或近似白色）的比例超过给定的阈值，则返回True，否则返回False。
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用一个阈值（例如200）来确定哪些像素是白色或近似白色的
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size

    if white_pixels / total_pixels > threshold:
        return True
    return False


def filter_slices(root_directory, threshold=0.99):
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            image_path = os.path.join(subdir, file)
            if is_empty_slice(image_path, threshold):
                os.remove(image_path)  # 删除该图像文件
                print(f"Removed: {image_path}")


# 使用函数
root_directory = r"/media/ubuntu/Elements SE/complete/"
filter_slices(root_directory)
