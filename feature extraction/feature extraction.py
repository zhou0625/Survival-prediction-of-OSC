import os
import cv2
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms

if torch.cuda.is_available():
    print('GPU')
    device = torch.device('cuda:0')
else:
    print('CPU')
    device = torch.device('cpu')

# 定义多尺度卷积块
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out = torch.cat((out1, out3, out5), dim=1)
        return out

# 定义Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
            )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.layer_norm1(x)
        out = self.attention(out, out, out)[0]
        out = self.dropout(out)
        out += residual

        residual = out
        out = self.layer_norm2(out)
        out = self.feed_forward(out)
        out = self.dropout(out)
        out += residual

        return out

# 定义多尺度卷积+Transformer模型
class MultiScaleConvTransformer(nn.Module):
    def __init__(self, img_size=512, in_channels=3):
        super(MultiScaleConvTransformer, self).__init__()
        self.conv_block = MultiScaleConvBlock(in_channels, 64) # 可以调整输出通道数
        self.avgpool = nn.AdaptiveAvgPool2d((64, 64))
        self.transformer_block = TransformerBlock(3 * 64, 8, 256, 0.1) # 调整输入维度
        self.fc = nn.Linear(3 * 64, 64) # 输出层

    def forward(self, x):
        out = self.conv_block(x)
        out = self.avgpool(out)
        out = out.permute(0, 2,3,1)
        out = out.view(out.size(0), 64*64, 192) # 展平特征图
        out = self.transformer_block(out)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out




preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



# 创建模型实例
model = MultiScaleConvTransformer()
model.to(device)
root_folder = r"/media/ubuntu/zzzr/2/"
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    if os.path.isdir(folder_path):  # 确保当前路径是文件夹而不是文件
        print("处理文件夹:", folder_name)
        file_path_list = []
        features = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            file_path_list.append(file_path)
            print("文件：", file_path)

        # 提取特征并保存为CSV
        for path in file_path_list:
            # 读取文件
            image = cv2.imread(path)
            # 预处理图像
            processed_image = preprocess(image).unsqueeze(0)
            processed_image = processed_image.to(device)
            # 将图像输入到模型中提取特征
            with torch.no_grad():
                feature = model(processed_image)

            # 将特征转换为NumPy数组并添加到特征列表中
            feature = feature.squeeze().cpu().numpy().tolist()
            features.append(feature)

            # 将特征保存为CSV文件
        output_path = os.path.join(folder_path, '{}.csv'.format(folder_name))

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for feature in features:
                writer.writerow(feature)