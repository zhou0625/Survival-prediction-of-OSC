import os
import time
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import metrics
import torchmetrics
from torch import optim, nn
from torch.nn import Conv2d, Dropout, Sequential, ReLU, MaxPool2d, Flatten
from torch.nn.modules import Linear, loss
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms



# 创建数据集
from torchvision.utils import save_image


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)

        # 使用cv2读取图片
        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 注意OpenCV使用BGR颜色模式
        #img = cv2.resize(img, (127, 127))

        # 转换为PIL Image对象以便使用transforms
        img = Image.fromarray(img)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5,))])
        img_tensor = transform(img)
        if self.label_dir == 'oc':
            label = 0
        else:
            label = 1

        label_Tensor = torch.from_numpy(np.array(label)).long()
        return img_tensor, label_Tensor

    def __len__(self):
        return len(self.img_path)


# 训练集
root_dir = r"G:\train-"
ok_labeldir = "ok"
oc_labeldir = "oc"
ok_traindata = MyData(root_dir, ok_labeldir)
oc_traindata = MyData(root_dir, oc_labeldir)
train_dataset = ok_traindata + oc_traindata

# 验证集
rootdir = r"G:\valid-"
ok_labeldir1 = "ok"
oc_labeldir1 = "oc"
ok_testdataset = MyData(rootdir, ok_labeldir1)
oc_testdataset = MyData(rootdir, oc_labeldir1)
test_dataset = ok_testdataset + oc_testdataset

# 数据集长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练集的长度为：{}".format(train_dataset_size))
print("测试集的长度为：{}".format(test_dataset_size))

# 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# 搭建神经网络
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model = Sequential(Conv2d(1, 32, 11, stride=1, padding=2),  # 32,127--121
                                ReLU(inplace=True),
                                Conv2d(32, 64, 7, stride=2, padding=1),  # 64,121--59
                                ReLU(inplace=True),
                                Conv2d(64, 32, 5, stride=2, padding=2),  # 32,59--30
                                ReLU(inplace=True),
                                #Conv2d(64, 32, 3, stride=1, padding=1),  # 16,32--32
                                #ReLU(inplace=True),
                                Conv2d(32, 16, 3, stride=1, padding=1),  # 16,30--30
                                ReLU(inplace=True),
                                MaxPool2d(2),
                                Flatten(),  # 16*16*16
                                Dropout(p=0.2, inplace=False),
                                Linear(3600, 64, bias=True),  # 4096=32*16*16
                                # ReLU(inplace=True),
                                Linear(64, 2, bias=True))
    def forward(self,x):
        x = self.model(x)
        return x
model = Mymodel()
#model = model.cuda()
print(model)

# 定义损失函数
loss_fun = loss.CrossEntropyLoss()
#loss_fun = loss_fun.cuda()
# 定义优化器
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 设置训练网络的参数
train_loss = []
val_loss = []
val_acc = []
epoch = 21
for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))
    # 训练开始
    total_train_step = 0
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        #imgs = imgs.cuda()
        #targets = targets.cuda()
        outputs = model(imgs)

        loss = loss_fun(outputs, targets)
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("train_step：{}，loss值：{}".format(total_train_step, loss.item()))
        # writer.add_scalar("train_loss", loss.item(), total_train_step

    total_loss = 0
    total_accuracy = 0
    total_test_step = 0
    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='none', num_classes=2)  # 两类的召回率
    test_precision = torchmetrics.Precision(average='macro', num_classes=2)  # 两类的平均/ 'micro'全部的平均

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            #imgs = imgs.cuda()
            #targets = targets.cuda()
            outputs = model(imgs)

            val_los = loss_fun(outputs, targets)
            val_los.requires_grad_(True)

            #prob = outputs.argmax(1).numpy()  # [:, 1]
            prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].numpy()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

            total_loss = total_loss + val_los.item()
            acc1 = total_accuracy/test_dataset_size
            total_test_step = total_test_step + 1

            # print(accuracy)

            test_acc(outputs.argmax(1), targets)
            test_recall(outputs.argmax(1), targets)
            test_precision(outputs.argmax(1), targets)
            # test_auc.update(outputs, targets)

            # roc = ROC(num_classes=2)
            fpr, tpr, threshold = metrics.roc_curve(targets, prob)
            # fpr, tpr, thresholds = roc(pre2, targets)
            roc_auc = metrics.auc(fpr, tpr)
            # roc_auc = test_auc.compute()

    # 计算一个epoch的accuray、recall、precision、AUC
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
# total_auc = test_auc.compute()
# print(prob)
# print(pre1)
    print("验证集上的loss值：{}".format(total_loss))
    print("验证集上的准确率：{}".format(total_accuracy / test_dataset_size))
    print(acc1)
    print(f"acc: ", (100 * total_acc))
    print("recall: ", total_recall)
    print("precision: ", total_precision)
# print("auc:", total_auc.item())

# 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    #test_auc.reset()


    val_acc.append(acc1.item())
    val_loss.append(val_los.item())
    train_loss.append(loss.item())
Loss1 = np.array(train_loss)
acc0 = np.array(val_acc)
Loss0 = np.array(val_loss)
np.save('./train_loss', Loss1)
np.save('./val_acc', acc0)
np.save('./val_loss', Loss0)
torch.save(model, "./256-new-model4-50".format(i))
print("模型已保存！")

plt.figure(figsize=(6, 6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label='Val AUC = %3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
