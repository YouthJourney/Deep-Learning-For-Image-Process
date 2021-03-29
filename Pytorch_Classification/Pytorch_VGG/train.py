# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2021/1/26 14:24 
SoftWare: PyCharm
"""
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from Pytorch_VGG.model import vgg
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), '../.'))
print(data_root)
image_path = data_root + '/data_set/flower_data/'
print(image_path)

train_dataset = datasets.ImageFolder(root=image_path + 'train', transform=data_transform['train'])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
print(flower_list)
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # indent的数值代表在字典的元素前缩进空格数
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_dataset = datasets.ImageFolder(root=image_path + 'val', transform=data_transform['val'])
val_num = len(val_dataset)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

model_name = 'vgg16'
net = vgg(model_name=model_name, num_class=5, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)

for epoch in range(30):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = ">" * int(rate * 50)
        b = "=" * int((1 - rate) * 50)
        print('\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}'.format(int(rate * 100), a, b, loss), end='')

    print()

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

        val_acc = acc / val_num
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_acc))

print("Finished Training")
