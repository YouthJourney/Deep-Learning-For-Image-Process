# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2020/11/13 20:13 
SoftWare: PyCharm
"""
import numpy as np
import torchvision
from torch import optim
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from test_pytorch_official_demo.model import LeNet

# 图像预处理
transform = transforms.Compose(
    [transforms.ToTensor(),  # 图像预处理方法打包， 将图像归一化,由[H, W, C]格式转化为[C, H, W]格式
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 图像标准化, 输出=（原始数据-均值）/ 标准差
)

# 50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)

# 将训练集导入进来，然后分成一个个step，每一step36，数据集是否打乱，载入数据的线程数，Windows系统下设置为0
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

# 10000张验证图片
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000, shuffle=False, num_workers=0)

val_data_iter = iter(val_loader)  # 将val数据转化成可迭代的迭代器
val_image, val_label = val_data_iter.next()  # 可以使用next()函数获取一批数据
# 元组类型，值不可改变
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 显示图片
# def imshow(img):
#     img = img / 2 + 0.5  # 反标准化, 值依然是0到1之间
#     npimg = img.numpy()  # 转化为numpy格式
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 由(c, h, w)-->(h, w, c)
#     plt.show()
#
#
# # print labels
# print(''.join('%7s' % classes[val_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(val_image))

net = LeNet()
# CrossEntropyLoss()损失函数里面已经包含softmax激活函数，所以在建立模型时最后的输出不需要用softmax函数
loss_function = nn.CrossEntropyLoss()
# net.parameters()所有可训练的参数，Adam为梯度优化函数
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):  # epoch将训练集迭代的轮次，这里将训练集迭代5次
    # 累加训练过程中的损失
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):  # 遍历训练集样本，返回每一轮训练集的data和step（也就是index）
        # 分离出图像和标签
        inputs, labels = data
        # 将历史损失梯度给清零，如果不清除历史梯度，就会对计算的历史梯度进行累加。
        optimizer.zero_grad()
        # 输入数据正向传播
        outputs = net(inputs)
        loss = loss_function(outputs, labels)  # 计算loss损失
        loss.backward()  # loss进行反向传播
        optimizer.step()  # 进行参数的更新

        # 打印训练过程，给程序员自己查看
        running_loss += loss.item()
        if step % 500 == 499:  # 每训练500step打印一次信息
            with torch.no_grad():  # 在接下来的计算过程中不要去计算每个结点的误差损失梯度
                outputs = net(val_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]  # 需要在维度1（第0个维度是batch）上找到最大值，[1]找到它对应的index，不需要最大值是多少
                accuracy = (predict_y == val_label).sum().item() / val_label.size(0)  # 测试样本的准确率

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))  # 得到这500个step中的平均训练损失和验证准确率

                running_loss = 0.0

print('Finished Training')

save_path = './lenet.pth'
torch.save(net.state_dict(), save_path)
