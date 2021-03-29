# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: model
DateTime: 2021/1/13 21:52 
SoftWare: PyCharm
"""
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()  # 解决多继承出现的冲突抑制问题
        self.conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(units=128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)   # input[batch, 28, 28, 1], output[batch, 26, 26, 32]
        x = self.flatten(x)  # output[batch, 26*26*32=21632
        x = self.d1(x)  # output[batch, 128]
        x = self.d2(x)  # output[batch, 10]

        return x
