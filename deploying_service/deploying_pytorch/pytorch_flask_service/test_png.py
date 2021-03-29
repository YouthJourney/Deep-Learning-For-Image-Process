# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: test_png
DateTime: 2021/3/16 11:49 
SoftWare: PyCharm
"""
from PIL import Image
import numpy as np

img = Image.open('./tulip.png')
# img.show()
img_array = np.array(img)
print(np.max(img_array))
for i in range(333):
    print(img_array[i])


# import cv2
#
# img1 = cv2.imread('test.png')
# cv2.imshow('src', img1)
# cv2.waitKey(0)
# img_array = np.array(img1)
# print(np.max(img_array))# # for i in range(333):
# #     for j in range(500):
# #         print(img_array[i][j])
