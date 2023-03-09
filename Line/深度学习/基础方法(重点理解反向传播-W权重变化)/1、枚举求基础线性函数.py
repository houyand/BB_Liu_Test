#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def forward(x):
    return x * w


def loss(x, y):
    return (y - x) * (y - x)


# 权重w  列表
w_list = []
# 权重w对应loss 列表
mse_list = []

if __name__ == '__main__':
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    for w in np.arange(0.0, 4.1, 0.1):
        print('w=', w)
        # 总的损失值
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)

            loss_val = loss(y_val, y_pred_val)
            l_sum += loss_val

            print('\t', x_val, y_val, y_pred_val, loss_val)
            # 平均损失值
            print('MSE=', l_sum / 3)
            w_list.append(w)
            mse_list.append(l_sum / 3)

# 画图
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
