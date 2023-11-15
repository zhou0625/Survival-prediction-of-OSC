import cv2
import numpy as np
import os

# 主目录路径
main_directory = r"C:\Users\86136\Desktop\1/"

# 获取所有子文件夹
subfolders = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

# 迭代每个子文件夹
for subfolder in subfolders:
    original_image_folder = os.path.join(subfolder, 'images/')
    mask_folder = os.path.join(subfolder, 'mask/')
    roi_folder = os.path.join(subfolder, 'extract/')

    # 获取原始图像和掩码的文件名
    original_image_files = [f for f in os.listdir(original_image_folder) if os.path.isfile(os.path.join(original_image_folder, f))]
    mask_files = [f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))]

    # 迭代原始图像和掩码
    for original_image_file, mask_file in zip(original_image_files, mask_files):
        original_image_path = os.path.join(original_image_folder, original_image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        # 读取原始图像和掩码
        original_image = cv2.imread(original_image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 使用掩码提取感兴趣区域
        #roi = cv2.bitwise_and(original_image, original_image, mask=mask)


        h, w, c = original_image.shape
        img3 = np.zeros((h, w, 4))
        img3[:, :, 0:3] = original_image
        img3[:, :, 3] = mask
        # 保存感兴趣区域到子文件夹
        output_path = os.path.join(roi_folder, original_image_file)
        cv2.imwrite(output_path, img3)


cv2.destroyAllWindows()
