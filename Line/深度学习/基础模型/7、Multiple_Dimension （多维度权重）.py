#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 多维输入


import torch
import torch.nn.functional as F  # sigmoid 函数
import numpy as np
import matplotlib.pyplot as plt


class MultipleDimension(torch.nn.Module):
    def __init__(self):
        super(MultipleDimension, self).__init__()
        # 神经网络多层模型，第一层输入8维，输出6维度
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()    #可以测试不同的激活函数对神经网络的影响
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))  # 每一次中间隐藏函数都要使用（sigmoid函数）做一次非线性变换
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))

        # x = self.relu(self.linear1(x))  # 每一次中间隐藏函数都要使用（sigmoid函数）做一次非线性变换
        # x = self.relu(self.linear2(x))
        # x = self.sigmoid(self.linear3(x))  #最后一层使用sigmoid  relu 如果小于0 输出0,计算ln0 就会有问题

        # 此处切换不同的激活函数可以看到loss的值是不一样的
        return x


if __name__ == '__main__':
    # -------------------第一步--------准备数据-------diabetes.csv 糖尿病数据集---------------------#
    xy = np.loadtxt('D:/学习资料/研究生/人工智能资料/PyTorch深度学习实践/diabetes.csv/diabetes.csv', delimiter=',', dtype=np.float32)
    print(xy)
    x_data = torch.from_numpy(xy[:, :-1])
    y_data = torch.from_numpy(xy[:, [-1]])

    # ---------------------第二步-----设计模型-----------------------------#
    model = MultipleDimension()
    # ----------------------第三步---------定义loss、优化器------------------------#
    criterion = torch.nn.BCELoss(size_average=True)    #注意BCELoss 数学模型  因为y-data是0-1，需要改一下模型
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  #model.parameters 更新模型中的每个权重参数

    # -------------------------第四步----训练模型--------------------------#
    for epoch in range(1000):
        y_pred = model(x_data)      #目前没有使用Dataload,不使用mini-batch  就是一次取一批数据进行训练，数据集很大的情况
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        #update 参数权重
        optimizer.step()
