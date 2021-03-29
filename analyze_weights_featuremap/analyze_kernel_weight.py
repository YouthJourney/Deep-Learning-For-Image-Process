import torch
from analyze_weights_featuremap.alexnet_model import AlexNet
from analyze_weights_featuremap.resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np

# create model
# model = AlexNet(num_classes=5)
model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./resNet34.pth"  # "resNet34.pth" # ./AlexNet.pth
model.load_state_dict(torch.load(model_weight_path))
print(model)

weights_keys = model.state_dict().keys()  # 获取所有具有参数的层结构名称
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:  # 排除BN层不必要的信息
        continue
    # [kernel_number, kernel_channel, kernel_height, kernel_width]
    weight_t = model.state_dict()[key].numpy()  # 读取key层的所有参数

    # read a kernel information
    # k = weight_t[0, :, :, :]  # 读取第一个卷积核的参数

    # calculate mean, std, min, max  # 计算该层所有卷积核的参数，均值、标准差、最小值、最大值
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1])  # 将卷积核的权重展成一个一维向量。
    plt.hist(weight_vec, bins=50)  # 使用hist方法来统计卷积核的权重值的一个分部，画直方图；
    # bins=50将所取的最小值和最大值的区间均分成50等份，然后再统计落到每一个小区间上的值的个数。
    plt.title(key)  # 给直方图加一个标题
    plt.show()
