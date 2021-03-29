# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: model
DateTime: 2021/1/20 16:38 
SoftWare: PyCharm
"""
from tensorflow.keras import layers, models, Model, Sequential


# 使用tf.keras模块搭建网络
def AlexNet_v1(im_height=224, im_width=224, class_num=1000):
    # tensorflow中的tensor通道排序默认是[N, H, W, C]
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')  # output[None, 224, 224, 3]
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # 图片填充 output[None, 227, 227, 3]
    x = layers.Conv2D(96, kernel_size=11, strides=4, activation='relu')(x)  # output[None, 55, 55, 96]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output[None, 27, 27, 96]
    x = layers.Conv2D(256, kernel_size=5, padding='same', activation='relu')(x)  # output[None, 27, 27, 256]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output[None, 13, 13, 256]
    x = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')(x)  # output[None, 13, 13, 384]
    x = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')(x)  # output[None, 13, 13, 384]
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)  # output[None, 13, 13, 256]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output[None, 6, 6, 256]

    x = layers.Flatten()(x)  # output[None, 6*6*256]
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)  # output[None, 2048]
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)  # output[None, 2048]
    x = layers.Dense(class_num)(x)  # output[None, class_num]

    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)

    return model


# 使用subclass的方法搭建网络，类似pytorch的网络搭建方法
class AlexNet_v2(Model):
    def __init__(self, class_num=1000):
        super(AlexNet_v2, self).__init__()
        self.feature = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),
            layers.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2)
        ])
        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(2048, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(2048, activation='relu'),
            layers.Dense(class_num),
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.feature(inputs),
        x = self.flatten(x)
        x = self.classifier(x)

        return x
