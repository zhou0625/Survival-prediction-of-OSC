# 首先将数据做出Dataset，并用Dataloader
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

import torchmetrics
from torch.nn.modules import loss
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Dropout
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 model 是你的模型，test_loader 是你的测试集的 DataLoader

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 注意OpenCV使用BGR颜色模式
        img = cv2.resize(img, (127, 127))
        img = Image.fromarray(img)
        #img = img.convert('RGB')
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

rootdir =r"G:\test-"
ok_labeldir = "ok"
oc_labeldir = "oc"
ok_testdataset = MyData(rootdir, ok_labeldir)
oc_testdataset = MyData(rootdir, oc_labeldir)
test_dataset = ok_testdataset + oc_testdataset

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
test_dataset_size = len(test_dataloader)

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model = Sequential(Conv2d(1, 32, 11, stride=1, padding=2),  # 32,127--63
                                ReLU(inplace=True),
                                Conv2d(32, 64, 7, stride=2, padding=1),  # 64,63--32
                                ReLU(inplace=True),
                                Conv2d(64, 32, 5, stride=2, padding=2),  # 32,61--32
                                ReLU(inplace=True),
                                Conv2d(32, 16, 3, stride=1, padding=1),  # 16,32--32
                                MaxPool2d(2),
                                ReLU(inplace=True),
                                Flatten(),  # 16*16*16
                                #Dropout(p=0.5, inplace=False),
                                Linear(3600, 64, bias=True),  # 4096=32*16*16
                                ReLU(inplace=True),
                                Linear(64, 2, bias=True))  # 16*16
    def forward(self,x):
        x = self.model(x)
        return x

#model = model.cuda()

model = torch.load(r"D:\PC project1\pytorchh\256-new-model4-50")
loss_fun = loss.CrossEntropyLoss()
test_accc = []
test_losss = []
test_precision = []
test_recall = []
test_f1 = []
# 假设 model 是你的模型，test_loader 是你的测试集的 DataLoader
  # 切换到评估模式，这将关闭dropout和batch normalization

all_predictions = []
all_labels = []
model.eval()
with torch.no_grad():  # 关闭梯度计算，我们不需要在测试阶段进行反向传播
    for images, labels in test_dataloader:

        outputs = model(images)  # 通过模型获得输出
        test_loss = loss_fun(outputs, labels)
        _, predicted = torch.max(outputs, 1)  # 获取预测类别
        #probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        #all_predictions.extend(probabilities)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算并打印各项指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)


fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC')
plt.legend(loc="lower right")
plt.show()


cm = confusion_matrix(all_labels, all_predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 Score: {}'.format(f1))

'''for i in range(epoch):

    test_recalll = torchmetrics.Recall(average='none', num_classes=2)  # 两类的召回率
    test_precisionn = torchmetrics.Precision(average='macro', num_classes=2)

    with torch.no_grad():
        for data in enumerate(test_dataloader):
            img, target = data
            test_pred = model(data)

            test_loss = loss_fun(test_pred, target)
            test_loss.requires_grad_(True)

            acc = (test_pred.argmax(1) == target)
            total_acc = total_acc + acc

            total_loss = total_loss + test_loss.item()

            test_acc = total_acc/test_dataset_size

            test_recalll(test_pred.argmax(1), target)
            test_precisionn(test_pred.argmax(1), target)

            total_test_step = total_test_step + 1
    print("验证集上的loss值：{}".format(total_loss))
    print("验证集上的准确率：{}".format(total_acc / test_dataset_size))

    test_accc.append(test_acc.item())
    test_losss.append(test_loss.item())
    test_recall.append(test_recalll.item())
    test_precision.append(test_precisionn.item())


acc0 = np.array(test_accc)
Loss0 = np.array(test_losss)
recall0 = np.array(test_recall)
precision0 = np.array(test_precision)

np.save('./test_acc', acc0)
np.save('./test_loss', Loss0)
np.save('./test_recall', recall0)
np.save('./test_precisions', precision0)'''


