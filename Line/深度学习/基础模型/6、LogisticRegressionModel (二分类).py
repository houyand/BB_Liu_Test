#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F  # sigmoid 函数
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 1个输入，1个输出

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


if __name__ == '__main__':
    # -------------------第一步--------准备数据----------------------------#
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    print(x_data)
    y_data = torch.Tensor([[0], [0], [1]])  # 分类
    # ---------------------第二步-----设计模型-----------------------------#
    model = LogisticRegressionModel()
    # ----------------------第三步---------定义loss、优化器------------------------#
    criterion = torch.nn.BCELoss(size_average=False)  # 注意BCELoss 数学模型
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters 更新模型中的每个权重参数

    # -------------------------第四步----训练模型--------------------------#
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('预测学习4个小时通过考试的概率：', 4.0, '模型预测通过考试的概率输出：', model(torch.tensor([[4.0]])).item()*100,'%')

    # -------------------------使用训练的模型做预测--------------------------#
    # x = np.linspace(0, 10, 200)
    # x_t = torch.Tensor(x).view((200, 1))  # 列表变为torch中tensor。200列，1行
    # y_t = model(x_t)
    # y = y_t.datasources.numpy()
    # plt.plot(x, y)
    # plt.plot([0, 10], [0.5, 0.5], c='r')
    # plt.xlabel('Hours')
    # plt.ylabel('Probability of Pass')
    # plt.grid()  # grid() 方法来设置图表中的网格线
    # plt.show()
