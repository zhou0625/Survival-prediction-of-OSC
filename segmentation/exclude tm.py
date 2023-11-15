import os
from PIL import Image

def image_transparency_ratio(img_path):
    """Return the ratio of transparent pixels to total pixels in the image."""
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        alpha = img.split()[-1]
        transparent_pixels = sum(1 for pixel in alpha.getdata() if pixel == 0)
        total_pixels = img.width * img.height
        return transparent_pixels / total_pixels
    return 0

def remove_highly_transparent_images(directory, threshold=0.6):
    """Remove images where the transparency ratio exceeds the given threshold."""
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                if image_transparency_ratio(file_path) > threshold:
                    print(f'Removing image with transparency ratio > {threshold}: {file_path}')
                    os.remove(file_path)
            except Exception as e:
                print(f'An error occurred with file {file_path}: {e}')

directory = r"/media/ubuntu/zzzr/2/" # 指定你想要扫描的文件夹路径
remove_highly_transparent_images(directory)
