from model.unettt import UNet
from torch.optim import lr_scheduler
from util.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_net(net, device, data_path, epochs=100, batch_size=32, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                               cooldown=0, min_lr=0, eps=1e-08)

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            for image, label in train_loader:
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)

                print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), './best_model(TCGA).pth')
                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)

            scheduler.step(loss)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = r"E:\output\TCGA-OV/"
    print("-----------------------------------------------------------------------------------------------------------")
    train_net(net, device, data_path, epochs=100, batch_size=32)
