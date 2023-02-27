#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 全是线性层的就是全连接网络
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)



# 测试
def test():
    correct = 0
    total = 0
    with torch.no_grad():   # 在这个里面的代码不会去计算梯度
        for data in test_loader:
            images, labels = data
            print(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)    #因为是矩阵运算：每一行是一个图片的预测情况，取出概率最大值就是对应的分类情况  dim表示沿着行代表一个图片样本
            total += labels.size(0)  #labels.size(0) 是batchsize 就是样本数据
            correct += (predicted == labels).sum().item()
            print('实际测试情况: %d %%' % (100 * correct / total))

# 训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
    # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 取300次数据运算数值再打印一次loss 的数值
    if batch_idx % 300 == 299:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
        running_loss = 0.0


# 此处对图片的处理还是将图片拉成一个向量来处理的，是非常原始的特征提取，后面体会使用卷积神经网络的方式
if __name__ == '__main__':
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),   #图片pillow  pil  变为 c*w*h 张量
        transforms.Normalize((0.1307,), (0.3081,))   #（均值，标准差）将每一个图片的像素值映射为 正太分布 ，这里的数值是前人计算出来的mnist数据  为什么要做标准化，因为神经网络喜欢计算0-1的数（x-U）/&
    ])

    train_dataset = datasets.MNIST(root='../dataset/mnist/',
                                   train=True,
                                   download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                  train=False,
                                  download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=batch_size)

    model=Net()
    # 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # 训练
    for epoch in range(10):
        train(epoch)
        test()
