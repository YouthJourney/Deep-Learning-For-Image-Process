# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: model
DateTime: 2020/11/13 20:01 
SoftWare: PyCharm
"""
import torch.nn as nn
import torch.nn.functional as F


# 定义模型
class LeNet(nn.Module):
    # 类的初始化函数
    def __init__(self):
        super(LeNet, self).__init__()  # 多层继承使用super
        # 按照CNN网络结构来建立层结构
        # 输入图像的深度为3，卷积核个数为16，卷积核尺寸5*5   N = (W-F+2P)/S + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        # 池化核大小2*2，步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 输入深度16，卷积核个数32，卷积核尺寸5*5
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 池化核大小2*2，步长为2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 将上一层输出的特征矩阵展平为一个一维向量，也就是32 * 5 * 5，由于全连接层的输入需要。结点个数为120
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # 输入就是上一层输出的120个结点，输出结点个数为84
        self.fc2 = nn.Linear(120, 84)
        # 输入就是上一层输出的84个结点，输出结点个数为10，需要根据训练集做修改，如本例的输出是10个类别因此输出个数为10
        self.fc3 = nn.Linear(84, 10)

    # 网络数据正向传播的一个过程
    # x代表输入的数据，即是Pytorch Tensor 的通道排序： [batch, channel, height, width]，对应于图片的张数、通道数（深度）、高度、宽度。
    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32); output(16, 28, 28)  28 = (32-5+0)/1 + 1
        x = self.pool1(x)  # output(16, 14, 14) 只改变特征矩阵的高和宽，深度不变
        x = F.relu(self.conv2(x))  # output(32, 10, 10)  10 = (14-5+0)/1 + 1
        x = self.pool2(x)  # output(32, 5, 5)
        # 使用view函数将特征矩阵展平为一维向量，-1代表第一个维度，第二个维度就是展平后的结点个数
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        # 内部已经使用了softmax激活函数将输出转化为概率分布，就不用添加softmax层。
        x = self.fc3(x)  # output(10)
        return x


# # 测试
# import torch
#
# input1 = torch.rand([32, 3, 32, 32])
# model = LeNet()
# print(model)
# output = model(input1)

