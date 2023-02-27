#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# torchversion.datasets 里面的数据集，也是实现dataset抽象类的
# MNIST
# • Fashion-MNIST
# • EMNIST
# • COCO
# • LSUN
# • ImageFolder
# • DatasetFolder
# • Imagenet-12
# • CIFAR
# • STL10
# • PhotoTour

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# 神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)   #输入8维,输出6维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # 前向传播,计算 pred
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))   #第一层计算数值 self.linear1(x)  第二层 sigmoid函数变换到0-1
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


if __name__ == '__main__':
    # train=True 要得是训练集还是测试数据集 true训练数据集
    train_dataset = datasets.MNIST(root='D:/pycharmworkspace/Torchversion_Datasets/mnist', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)  # transform= transforms.ToTensor() 是将图像里面的PIL 转化为Tensor
    test_dataset = datasets.MNIST(root='D:/pycharmworkspace/Torchversion_Datasets/mnist', train=False,
                                  transform=transforms.ToTensor(), download=True)  # download=True 设置如果没有这个数据集是否需要下载

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # model = Model()
    # criterion = torch.nn.BCELoss(size_average=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    # for batch_idx, (inputs, target) in enumerate(train_loader):


