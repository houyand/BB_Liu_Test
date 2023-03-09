#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt


# 预测
def forward(x):
    return x * w


# 求mse loss
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# 求loss 函数的偏导，也就是梯度
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)     #根据设计的loss函数计算值去算梯度
    return grad / len(xs)

# echo 循环次数
w_list = []
# 每一轮w 对应loss 列表
mse_list = []

if __name__ == '__main__':
    w = 0.1
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    print('Predict (before training)', 4, forward(4))
    for epoch in range(100):
        # 平均损失
        cost_val = cost(x_data, y_data)
        w_list.append(epoch)   #打印行数据
        mse_list.append(cost_val) #打印列数据
        grad_val = gradient(x_data, y_data)
        w -= 0.01 * grad_val
        print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
    print('Predict (after training)：', 4, 'result:',forward(4))

#画图
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
