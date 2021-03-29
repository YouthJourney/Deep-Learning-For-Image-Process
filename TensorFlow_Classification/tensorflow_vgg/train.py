# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: train
DateTime: 2021/1/26 16:03 
SoftWare: PyCharm
"""
import matplotlib.pyplot as plt
from tensorflow_vgg.model import vgg
import tensorflow as tf
import json
import os
import time
import glob
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)

data_root = os.path.abspath(os.path.join(os.getcwd(), "../."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
val_dir = image_path + "val"

# create direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

# class dict
data_class = [cla for cla in os.listdir(train_dir) if '.txt' not in cla]
print(data_class)
class_num = len(data_class)
class_dict = dict((val, index) for index, val in enumerate(data_class))
print(class_dict)

# reverse value and key of dict
inverse_dict = dict((val, key) for key, val in class_dict.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# load train images list
train_image_list = glob.glob(train_dir + "/*/*.jpg")  # 遍历所有图片
random.shuffle(train_image_list)  # 打乱所有图片
train_num = len(train_image_list)  # 训练集数量
train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]

# load validation images list
val_image_list = glob.glob(val_dir + '/*/*.jpg')
random.shuffle(val_image_list)
val_num = len(val_image_list)
val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]


def process_path(img_path, label):
    label = tf.one_hot(label, depth=class_num)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [im_height, im_width])

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE

# load train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_dataset = train_dataset.shuffle(buffer_size=train_num)\
    .map(process_path, num_parallel_calls=AUTOTUNE)\
    .repeat().batch(batch_size).prefetch(AUTOTUNE)

# load val dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .repeat().batch(batch_size)


# 实例化模型
model = vgg(model_name='vgg16', im_height=im_height, im_width=im_width, class_num=class_num)
model.summary()

# using keras high level api for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['acc'],
)

callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath='./save_weights/myVGG_{epoch}.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss'
)]

history = model.fit(
    x=train_dataset,
    steps_per_epoch=train_num // batch_size,
    epochs=epochs,
    validation_data=val_dataset,
    validation_steps=val_num // batch_size,
    callbacks=callbacks
)


