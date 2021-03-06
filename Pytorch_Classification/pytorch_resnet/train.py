# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2021/2/3 21:06 
SoftWare: PyCharm
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from pytorch_resnet.model import resnet34, resnet101
import torchvision.models.resnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # 原图片的长宽比不变，通过resize将最小边缩放到256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), '../.'))
image_path = data_root + '/data_set/flower_data/'
print(image_path)

train_dataset = datasets.ImageFolder(
    root=image_path + 'train',
    transform=data_transform['train']
)

train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_dataset = datasets.ImageFolder(
    root=image_path + 'val',
    transform=data_transform['val']
)
val_num = len(val_dataset)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)


net = resnet34()  # 没有传入num_class, 最后的输出是1000种分类
# load pretrain weights
model_weight_path = './resnet34-pre.pth'
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# 方法1（可以在定义时，就传入分类数5，然后将全连接层的参数给删掉。
# for param in net.parameters():
#     param.requires_grad = False
# 方法2
# change fc layer structure
in_channel = net.fc.in_features  # 获得原ResNet34网络的fc层的输入特征矩阵深度
net.fc = nn.Linear(in_channel, 5)  # 重新定义fc层。   这种载入权重的方法，也是官方提供的一种方法
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './ResNet34.pth'
for epoch in range(3):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
