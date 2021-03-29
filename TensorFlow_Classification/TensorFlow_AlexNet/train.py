# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2021/1/20 17:19 
SoftWare: PyCharm
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from TensorFlow_AlexNet.model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
image_path = data_root + '/Pytorch_Projects/data_set/flower_data/'
train_dir = image_path + 'train'
val_dir = image_path + 'val'

if not os.path.exists('save_weights'):
    os.mkdir('save_weights')

im_height = 224
im_width = 224
batch_size = 32
epochs = 6

# data generator with data augmentation
# 使用ImageDataGenerator函数会把label转化为one-hot编码格式
train_image_generator = ImageDataGenerator(
    rescale=1. / 255,  # 缩放，归一化
    horizontal_flip=True  # 水平翻转
)
val_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    shuffle=True,
    target_size=(im_height, im_width),
    class_mode='categorical'
)
# 训练集数目
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices
print('class_indices', class_indices)
# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = val_image_generator.flow_from_directory(
    directory=val_dir,
    batch_size=batch_size,
    shuffle=False,
    target_size=(im_height, im_width),
    class_mode='categorical'
)
# 验证集数目
total_val = val_data_gen.n

# display the picture
# sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding
# #
# #
# #  This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
#
# plotImages(sample_training_images[:5])


# 第一种搭建模型的方法训练
model = AlexNet_v1(im_height=im_height, im_width=im_width, class_num=5)
model.summary()

# using keras high level api for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 搭建模型进行了softmax()处理就是False， 没有的话就是TRUE
    metrics=['acc']
)

callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath='save_weights/AlexNet.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss'  # 监控的参数
)]

# TensorFlow2.1版本后， recommend to using fit, 它实现了在训练时打开Dropout，在验证时关闭Dropout方法
history = model.fit(
    x=train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=callbacks
)

# plot loss and accuracy image
history_dict = history.history
train_loss = history_dict['loss']
train_acc = history_dict['acc']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_acc']

# figure 1
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# figure 2
plt.figure()
plt.plot(range(epochs), train_acc, label='train_acc')
plt.plot(range(epochs), val_acc, label='val_acc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

# # 第二种搭建模型的方法训练
# model = AlexNet_v2(class_num=5)
# model.build((batch_size, im_height, im_width, 3))  # 只有在调用build时它才是真正的实例化了模型


# tensorflow2.1以前的版本若数据不能一次性加载到内存，需要使用fit_generator()方法进行训练
# history = model.fit_generator(generator=train_data_gen,
#                               steps_per_epoch=total_train // batch_size,
#                               epochs=epochs,
#                               validation_data=val_data_gen,
#                               validation_steps=total_val // batch_size,
#                               callbacks=callbacks)


# using keras low level api for training
# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
#
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
#
#
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = loss_object(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
#
# @tf.function
# def test_step(images, labels):
#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#
#
# best_test_loss = float('inf')
# for epoch in range(1, epochs+1):
#     train_loss.reset_states()        # clear history info
#     train_accuracy.reset_states()    # clear history info
#     test_loss.reset_states()         # clear history info
#     test_accuracy.reset_states()     # clear history info
#     for step in range(total_train // batch_size):
#         images, labels = next(train_data_gen)
#         train_step(images, labels)
#
#     for step in range(total_val // batch_size):
#         test_images, test_labels = next(val_data_gen)
#         test_step(test_images, test_labels)
#
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#     print(template.format(epoch,
#                           train_loss.result(),
#                           train_accuracy.result() * 100,
#                           test_loss.result(),
#                           test_accuracy.result() * 100))
#     if test_loss.result() < best_test_loss:
#        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')
