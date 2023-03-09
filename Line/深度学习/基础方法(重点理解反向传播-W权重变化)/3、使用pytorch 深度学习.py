#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt


# 函数，也可以说是需要的模型，后面继续优化，模型也不手动固定
def forward(x):
    return x * w


# 求loss
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# echo 循环次数
w_list = []
# 每一轮w 对应loss 列表
loss_list = []

if __name__ == '__main__':
    w = torch.Tensor([1.0])  # Pytorch 基本数据成员 Tensor 用来存数据，可以是标量，向量，矩阵，里面包含data ，是w的数值,grad：loss对w求梯度下降后的结果
    w.requires_grad = True  # 设置需要计算梯度

    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    print('Predict (before training)', 4, forward(4).item())
    for epoch in range(100):
        w_list.append(epoch)
        for x, y in zip(x_data, y_data):
            loss_data = loss(x, y)
            loss_data.backward()    #会自动计算所有链路上的梯度，计算的最后结果保存在 w.grad 这里。
            print('\tgrad:', x, y, w.grad.item())   #.item() 是取的数值
            w.data = w.data - 0.01 * w.grad.data   #注意grad 也是tensor ,需要取grad的数值data参与运算，结果还是tensor

            #上一次w 的梯度数值 清零
            w.grad.data.zero_()
            print('Epoch:', epoch, 'w=', w.data.item(), 'loss=', loss_data.item())

        loss_list.append(loss_data.item())
    print('Predict (after training)', 4, forward(4).item())

# 画图
plt.plot(w_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
