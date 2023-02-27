#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:/test.jpeg', 0)
# 图像经过傅里叶变换(fft2)得到 f (复数的数组)
f = np.fft.fft2(img)
# 将低频数组由左上角移动到中间窗口
fshift = np.fft.fftshift(f)

# 逆傅里叶变换-低频由中心变为左上角
ishift=np.fft.ifftshift(fshift)
# 由频谱变为实数-但是含有正负数
io=np.fft.ifft2(ishift)
# 取绝对值
io=np.abs(io)



# 显示原始图像
plt.subplot(121)
plt.imshow(img, cmap='gray')   #cmap显示灰度图像
plt.title('original')
plt.axis('off')    #表示不要坐标轴


plt.subplot(122)
plt.imshow(io, cmap='gray')   #cmap显示灰度图像
plt.title('result')
plt.axis('off')    #表示不要坐标轴

plt.show()