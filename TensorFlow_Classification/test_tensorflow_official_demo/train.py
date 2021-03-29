# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2021/1/13 22:11 
SoftWare: PyCharm
"""
import tensorflow as tf
from test_tensorflow_official_demo.model import MyModel

# tf.keras.backend.set_floatx('float64')
mnist = tf.keras.datasets.mnist

# download and load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# # view images
# imgs = x_test[:3]
# labs = y_test[:3]
# print(labs)
# import matplotlib.pyplot as plt
# import numpy as np
# plt_imgs = np.hstack(imgs)
# plt.imshow(plt_imgs, cmap='gray')
# plt.show()

# Add a channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)
# create data generator
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# create model
model = MyModel()

# model.summary()

# define loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# define optimizer
optimizer = tf.keras.optimizers.Adam()

# define train_loss and train_accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')  # 平均训练损失值
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

# define test_loss and test_accuracy
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')


# define train function including calculating loss, applying gradient and calculating accuracy
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# define test function including calculating loss and calculating accuracy
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# train process
EPOCH = 10

for epoch in range(EPOCH):
    # 将每一轮的损失和准确率累加器都重置
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%.'
    print(template.format(
        epoch + 1,
        train_loss.result(),
        train_accuracy.result() * 100,
        test_loss.result(),
        test_accuracy.result() * 100
    ))

model.summary()
