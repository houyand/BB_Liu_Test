#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F  # 使用rulu 这个激活函数  0-1


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        # 此处的320 是由于视频里面使用28*28的图像，最后得到是20*4*4 得到320。重要，最后这个全连接层的数值要么手算，要么先别定义，使用pytorch算了之后再设置
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
    # Flatten datasources from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # 变成全连接想要的网络 就是一个长条
        x = self.fc(x)    #后面继续做交叉熵损失，就不需要进行relu了
        return x


if __name__ == '__main__':

    model = Net()
    # #使用GPU计算 配置
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #cuda:0 是可以调整的，根据显卡的数量来
    # #这样通过torch的to() 方法就把计算方式设置为了Gpu 另外一个需要改的地方就是数据与标签的地方inputs, target = inputs.to(device), target.to(device)
    # model.to(device)
    in_channels, out_channels = 5, 10
    width, height = 100, 100
    kernel_size = 3   #卷积核大小
    batch_size = 1
    input = torch.randn(batch_size,
                        in_channels,
                        width,
                        height)
    # 卷积核  此处10个通道channel 都是共用一样权重的卷积核,这里的卷积核权重是随机生成的
    conv_layer = torch.nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=kernel_size)
    output = conv_layer(input)
    print(input)
    print(output.shape)
    print(conv_layer.weight.shape)  # torch.Size([10, 5, 3, 3])  10 输出通道数量  5输入通道数量

    # 第二个测试 padding=1 表示在原图像层外加了一圈0，这样3*3卷积下来后还是一样的大小，如果是5*5的卷积核，则padding=5/2=2 （这样去计算）
#     input = [3, 4, 6, 5, 7,
#              2, 4, 6, 8, 2,
#              1, 6, 7, 8, 4,
#              9, 7, 4, 6, 2,
#              3, 7, 5, 4, 1]
#     print('初始input:\n',input)
#     input = torch.Tensor(input).view(1, 1, 5, 5)  #viem 将input向量变成了batch为1,channel为1，weight为5，height为5
#     print('转化化后input:\n',input)
#
#     # 模型就是卷积核怎么定义
#     conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#     # conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2,bias=False)
#     # 自定义卷积核的初始化值
#     kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
#     conv_layer.weight.datasources = kernel.datasources
#
#     output = conv_layer(input)
#     print(output)
#
#
# #    下采样，或者说是池化
#     maxpooling_layer=torch.nn.MaxPool2d(kernel_size=2)
#     print(maxpooling_layer(output))
