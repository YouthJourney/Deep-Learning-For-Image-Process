# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: predict
DateTime: 2021/1/20 16:01 
SoftWare: PyCharm
"""
import torch
from pytorch_alexnet.model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load image
img = Image.open('tulip.jpg')
plt.imshow(img)
plt.show()

# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('class_indices.json')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = 'AlexNet.pth'
model.load_state_dict(torch.load(model_weight_path))
model.eval()  # 关闭Dropout方法
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))  # 将正向传播得到的结果压缩掉batch这个维度
    predict = torch.softmax(output, dim=0)  # 结果变成一个概率分布
    predict_cla = torch.argmax(predict).numpy()  # 找到概率分布最大值所对应的索引值
    print(predict_cla)

print(class_indict[str(predict_cla)], predict[predict_cla].item())  # 打印类别名称和对应的概率

