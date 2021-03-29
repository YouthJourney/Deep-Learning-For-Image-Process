# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: predict
DateTime: 2021/1/26 15:36 
SoftWare: PyCharm
"""
import torch
from Pytorch_VGG.model import vgg
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from PIL import Image

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load image
img = Image.open('tulip.jpg')
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = vgg(model_name="vgg16", num_classes=5)
# load model weights
model.load_state_dict(torch.load('vgg16Net.pth'))
model.eval()
with torch.no_grad():
    # predict class
    outputs = torch.squeeze(model(img))
    predict = torch.softmax(outputs, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)])
plt.show()
