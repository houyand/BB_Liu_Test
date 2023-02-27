#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Module 里面自动实现 backward()    在基础方法上继续使用pytorch
import torch

# 完全使用pytorch 设计模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象 1个权重参数，1个输出

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

#一次训练一批数据
if __name__ == '__main__':
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[2.0], [4.0], [6.0]])

    model = LinearModel()

    criterion = torch.nn.MSELoss(size_average=False)   #loss 标准  是否需要计算平均loss size_average
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  #优化器  模型每个参数权重w

    for epoch in range(1000):
        y_pred = model(x_data)              #1、模型预测
        loss = criterion(y_pred, y_data)    #2、计算loss
        print(epoch, loss.item())
        optimizer.zero_grad()               #3、优化器先归零,也就是调整权重先归零，下一步开始计算
        loss.backward()                     #4、反向传播，计算调整梯度
        optimizer.step()                    #5、优化调整模型中每个参数的权重W

# Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data.item())
