import os
from PIL import Image

count = 0
def read_path(file_pathname):# 函数的输入是图片所在的路径
    count = 0
    for filename in os.listdir(file_pathname):
        img = Image.open(file_pathname+'/'+filename)            # 读取文件
        im_rotate = img.rotate(90)                              #图像旋转
        im_rotate.save(r"F:\train\oc-augment" + '/' + 'oc-90-{}.png'.format(count))# 图片保存
        count += 1
read_path(r"F:\train\oc/")
count += 1

