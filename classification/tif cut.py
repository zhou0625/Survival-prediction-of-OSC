import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from PIL import Image

os.add_dll_directory(r'C:\Users\86136\anaconda3\openslide-win64-20220811\bin')
import openslide

Image.MAX_IMAGE_PIXELS = None

# imgdir = r"E:\TCGA-pathology"
# img = openslide.open_slide(r"E:\ovary-tif\OV-Platinum resistance\5-4 14.3560-3-HE_Default_Extended.tif")
# 原始图像文件夹路径
img_path = r"F:\test-normal/"

# 原始图像文件列表
file_path_list = []

# 获取文件夹下所有文件名（请确保不存在子目录）
for file in os.listdir(img_path):
    file_path = os.path.join(img_path, file)
    file_path_list.append(file_path)
    print("文件：", file_path)

# ROI图像保存路径
imgsavedir = r"F:\test-normal/"

# print(type(img))3
resx = 256
resy = 256
for path in file_path_list:
    # 读取文件
    img = openslide.open_slide(path)
    [m, n] = img.dimensions
    # 读取文件名（更换原始图像文件夹路径时此处数值需更改）
    file_name = path[17:-4]
    # 在ROI图像保存路径生成新文件夹，以图像名称命名
    save_dir = imgsavedir + "\\" + file_name
    # 判断文件夹是否已存在，若无则新建
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    count = 1
    for i in range(m // resx):
        for j in range(n // resy):
            roi = np.array(img.read_region((i * resx, j * resy), 0, (resx, resy)))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # 将roi保存到指定文件夹，并以原名称命名
            cv2.imwrite(save_dir + "\\" '{}.png'.format(count), roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #cv2.imwrite(imgsavedir + "\\" '{}.png'.format(count), roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            count += 1
