# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: model
DateTime: 2021/1/18 21:11 
SoftWare: PyCharm
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # 使用Sequential简化代码
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            # input[3, 224, 224], output[96, 55, 55]  N = (W - F + 2P)/S + 1, 55 = (224-11 + (1+2))/4 + 1
            nn.ReLU(inplace=True),  # inplace=Ture通过一种方法增加计算量，但是会降低内存使用，以便在内存中载入更大的模型。
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # output[256, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 使结点随机失活一半
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

        if init_weights:  # 当前版本中默认使用kaiming方法进行初始化权重，不用手动去初始化
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(input=x, start_dim=1)  # 在经过第三步池化之后，主要将数据进行展平，故而添加flatten层, 从channel维度开始展平，亦可使用view()函数进行展平
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历网络的每一个层结构
            if isinstance(m, nn.Conv2d):  # 判别层结构是否为卷积层，若是，就用kaiming初始化权重；后面若偏置不为空，就将偏置置为0
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何凯明提出
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 若为全连接层，就采用一个正太分布给权重进行赋值。
                nn.init.normal_(m.weight, 0, 0.01)  # 均值为0， 方差为0.01
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
