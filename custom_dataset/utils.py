# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: utils
DateTime: 2021/3/17 11:06 
SoftWare: PyCharm
"""
import os
import json
import pickle
import random

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机过程可复现，也就是每次生成的随机数相同。
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    print('flower_class: ', flower_class)
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_image_path = []  # 存储训练集的所有图片路径
    train_image_label = []  # 存储训练集对应的索引信息
    val_image_path = []  # 存储验证集的所有图片路径
    val_image_label = []  # 存储每个类别的样本总数

    every_class_num = []  # 存储每个类别的样本总数
    supported = ['.jpg', '.JPG', '.png', '.PNG']  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        # 遍历每个类别文件夹下的所有文件，找出支持的文件格式文件，将root，类别名，文件名拼接形成图片的文件名。
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径采样的是验证集样本则存入验证集
                val_image_path.append(img_path)
                val_image_label.append(image_class)
            else:  # 否则存入训练集
                train_image_path.append(img_path)
                train_image_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))

    plot_image = False  # 是否绘制每个类别的图片数量柱状图
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0，1，2，3，4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_image_path, train_image_label, val_image_path, val_image_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + 'does not exist.'
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H , W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)  # 调整通道顺序
            # 反Normalization操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
