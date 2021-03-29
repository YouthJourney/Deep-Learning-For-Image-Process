# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: predict
DateTime: 2020/11/13 20:13 
SoftWare: PyCharm
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from test_pytorch_official_demo.model import LeNet
# 加载保存的模型权重，来进行图片预测

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图片形状变成和之前的一致
    transforms.ToTensor(),  # 归一化和通道顺序改变
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))  # 加载保存的权重

img = Image.open('plane.jpg')
img = transform(img)
img = torch.unsqueeze(img, dim=0)  # [N, C, H, W] 在最前面增加一个维度
# print(img)

with torch.no_grad():
    outputs = net(img)
    predict = torch.max(outputs, dim=1)[1].data.numpy()

print(classes[int(predict)])

# with torch.no_grad():
#     outputs = net(img)
#     predict = torch.softmax(outputs, dim=1)  # 使用softmax处理得到的结果
#
# print(predict)
