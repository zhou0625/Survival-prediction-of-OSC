from PIL import Image
import os
import os.path

# rootdir 是被遍历的文件夹
rootdir = r"F:\train\oc/"

for parent, dirnames, filenames in os.walk(rootdir):  # 遍历图片
    count = 1
    for filename in filenames:

        currentPath = os.path.join(parent, filename)

        im = Image.open(currentPath)
        out = im.transpose(Image.FLIP_LEFT_RIGHT)  # 实现左右翻转
        #out = im.transpose(Image.FLIP_TOP_BOTTOM)#实现上下翻转
        newname = r"F:\train\oc-augment/" + '\\' + 'oc-y-{}.png'.format(count)  # 重新命名
        count += 1
        out.save(newname)  # 保存结束
