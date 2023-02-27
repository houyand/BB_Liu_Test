#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt


# 函数，也可以说是需要的模型，后面继续优化，模型也不手动固定
def forward(x):
    return x ** 2 * w1 + x * w2 + b


# 求loss
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# echo 循环次数
w_list = []
# 每一轮w 对应loss 列表
loss_list = []

if __name__ == '__main__':
    w1 = torch.Tensor([1.0])  # Pytorch 基本数据成员 Tensor 用来存数据，可以是标量，向量，矩阵，里面包含data w的值,grad：loss对w求导
    w2 = torch.Tensor([1.0])
    b = torch.Tensor([1.0])

    w1.requires_grad = True  # 设置需要计算梯度
    w2.requires_grad = True  # 设置需要计算梯度
    b.requires_grad = True  # 设置需要计算梯度

    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    print('Predict (before training)', 4, forward(4).item())
    for epoch in range(100):
        w_list.append(epoch)
        total_loss = 0
        for x, y in zip(x_data, y_data):

            loss_data = loss(x, y)
            total_loss += loss_data
            loss_data.backward()  # 会自动计算所有链路上的梯度，计算的最后结果保存在 w

            print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())  # loss 对w1,w2,b 梯度值

            w1.data = w1.data - 0.01 * w1.grad.data
            w2.data = w2.data - 0.01 * w2.grad.data
            b.data = b.data - 0.01 * b.grad.data

            # 上一次w 的梯度数值 清零
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            b.grad.data.zero_()

        total_loss =(total_loss / len(x_data)).item()
        loss_list.append(total_loss)
        print('Epoch:', epoch, 'w1=', w1.data.item(), 'w2=', w2.data.item(), 'b=', b.data.item(), 'loss=', total_loss)


    print('Predict (after training)', 4, forward(4).item())

# 画图
plt.plot(w_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
