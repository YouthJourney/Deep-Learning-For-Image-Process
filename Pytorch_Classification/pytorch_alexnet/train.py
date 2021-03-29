# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2021/1/19 21:36 
SoftWare: PyCharm
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from pytorch_alexnet.model import AlexNet
import os
import json
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 在水平方向上随机翻转
        transforms.ToTensor(),  # 归一化，转换通道顺序，也就是tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # cannot 224, must(224, 224)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))  # get data root path
print(data_root)
image_path = data_root + '/Pytorch_Projects/data_set/flower_data'  # flower data set path
print(image_path)
train_dataset = datasets.ImageFolder(root=image_path + '/train', transform=data_transform['train'])
train_num = len(train_dataset)
print(train_num)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} 雏菊、蒲公英、玫瑰、向日葵、郁金香
flower_list = train_dataset.class_to_idx  # 获取每个分类对应的索引值
print(flower_list)
cla_dict = dict((val, key) for key, val in flower_list.items())  # 将K，V反过来。

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # 将字典编码为json格式
with open('class_indices.json', 'w') as json_file:  # 写入文件
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # 加载数据所使用的线程数，在Windows下不能使用非0数，代表使用主线程去加载数据。
)

validate_dataset = datasets.ImageFolder(
    root=image_path + '/val',
    transform=data_transform['val']
)

val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(
    validate_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# display the picture
# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=5, init_weights=True)
net.to(device)  # 将网络指定到特定的设备上执行
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())  # 查看模型参数
optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 学习率为测试所得

save_path = 'AlexNet.pth'
best_acc = 0.0  # 为了得到准确率比较高的模型

for epoch in range(10):
    # train
    net.train()  # 管理Dropout层和Batch Normalization的，是这些操作只在训练时起作用，验证时不起作用。
    running_loss = 0.0
    t1 = time.perf_counter()  # 训练一个epoch所需要的时间
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()  # 清除历史梯度
        outputs = net(images.to(device))  # 将训练对象指定到特定的设备当中。
        loss = loss_function(outputs, labels.to(device))  # 求得损失。
        loss.backward()  # 将得到的损失反向传播到每一个节点当中。
        optimizer.step()  # 再通过optimizer更新每一个节点的参数。

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)  # 训练进度
        a = '*' * int(rate * 50)
        b = '*' * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():  # 禁止对参数进行跟踪
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()  # 得到验证集中预测正确的样本个数
        val_accurate = acc / val_num  # 得到验证集准确率
        if val_accurate > best_acc:
            best_acc = val_accurate  # 得到最优的验证准确率
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))


print('Finished Training')
