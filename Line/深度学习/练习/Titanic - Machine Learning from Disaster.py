#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Dataset实现类，加载数据，实现抽象类Dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  #shape 拿到一个矩阵的行列统计 [9,8]
        self.x_data = torch.from_numpy(xy[:, :-1])   #取数据除最后一列外前面所有列
        self.y_data = torch.from_numpy(xy[:, [-1]])  #取最后一列数据并转化为向量

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.linear2 = torch.nn.Linear(8, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # 前向传播,计算 pred
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))   #第一层计算数值 self.linear1(x)  第二层 sigmoid函数变换到0-1
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


if __name__ == '__main__':
    dataset = DiabetesDataset('D:/pycharmworkspace/Datas/titanic/train.csv')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  #shuffle：就是将数据集数据打乱重组 num_workers：2个线程

    print(train_loader)

    model = Model()
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        # enumerate() 函数将traim_loader 数据下标组合起来类似这[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        #0应该表示的是从下标从 0 计算
        for i, data in enumerate(train_loader, 0):
            # 1. Prepare data
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()
