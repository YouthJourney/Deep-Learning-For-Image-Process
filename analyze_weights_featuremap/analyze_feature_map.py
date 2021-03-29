import torch
from analyze_weights_featuremap.alexnet_model import AlexNet
from analyze_weights_featuremap.resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# data_transform = transforms.Compose(
#     [transforms.Resize((224, 224)),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
# model = AlexNet(num_classes=5)
model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./resNet34.pth"  # "./resNet34.pth, ./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open("./tulip.jpg")
# [N, C, H, W]

img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        # [H, W, C]
        # 特征矩阵的每一个channel所对应的是一个二维特征矩阵，就像灰度图像一样，channel=1
        plt.imshow(im[:, :, i])  # cmap='gray'如果不指定的话它是使用蓝色和绿色来替代我们灰度图像的黑色和白色。
    plt.show()

"""
上面代码是显示第0层、第3层和第六层卷积层输出的特征图，显示每个输出特征层的前12个channel的特征图，
可以发现在越往后的输出特征层里面，输出的特征图更加的抽象，还有很多是黑色的，说明这些卷积核没有起作用，
没有学到特征。

在使用ResNet网络时，明显可以看到网络学习到的信息要比AlexNet网络要多。
"""