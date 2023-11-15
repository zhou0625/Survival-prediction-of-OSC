import glob
import numpy as np
import torch
import os
import cv2
from model.unettt import UNet
from PIL import Image
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)
    # 加载网络，图片单通道，分类为1。
    net =UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(r'D:\PC project1\best_model(TCGA).pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径


    # 指定大文件夹路径
    root_folder = r"E:\1/"

    # 遍历大文件夹下的每个子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):  # 确保当前路径是文件夹而不是文件
            print("处理文件夹:", folder_name)

            # 遍历子文件夹下的图片文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):  # 确保当前路径是文件而不是文件夹
                    # 检查文件扩展名是否为图片格式，例如.jpg、.png等
                    if file_path.endswith(('.png')):
                        print("处理图片:", file_name)
                        # 在这里可以对每个图片文件进行进一步的处理
                        # 例如，使用PIL库加载和处理图像
                        #image = Image.open(file_path)
                        # 进行图像处理操作
                        # ...

                        #tests_path = glob.glob(r'G:\1-10\TCGA-3P-A9WA-01Z-00-DX1/*.png')
                        # 遍历素有图片
                        #for test_path in tests_path:
                        # 保存结果地址
                        #save_res_path = test_path.split('.')[0] + '_res.png'


                        # 读取图片
                        img = cv2.imread(file_path)
                        origin_shape = img.shape
                        print(origin_shape)
                        # 转为灰度图
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = cv2.resize(img, (512, 512))
                        # 转为batch为1，通道为1，大小为512*512的数组
                        img = img.reshape(1, 1, img.shape[0], img.shape[1])
                        # 转为tensor
                        img_tensor = torch.from_numpy(img)
                        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
                        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
                        pred = net(img_tensor)
        # 提取结果
                        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
                        pred[pred >= 0.5] = 255
                        pred[pred < 0.5] = 0
        # 保存图片
                        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
                        output_path = os.path.join(folder_path, f"mask_{file_name}")

                        cv2.imwrite(output_path, pred)
