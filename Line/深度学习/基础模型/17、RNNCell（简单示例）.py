#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


if __name__ == '__main__':
    input_size = 4
    hidden_size = 4
    batch_size = 1

    # 准备输入数据，准备输出结果。多分类任务
    idx2char = ['e', 'h', 'l', 'o']  # 词编码0-e,1-h,2-l,3-o
    x_data = [1, 0, 2, 2, 3]  # input:     Hello
    y_data = [3, 1, 2, 3, 2]  # output:    ohlol

    # 词向量模型[1, 0, 0, 0] 表示e,[0, 1, 0, 0]表示h
    one_hot_lookup = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # 构建输入的词向量
    x_one_hot = [one_hot_lookup[x] for x in x_data]
    print('x_data词向量：', x_one_hot)
    inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)

    print("inputs:", inputs)
    print("inputs-shape:", inputs.shape)

    # y_hot = [one_hot_lookup[x] for x in y_data]
    # labels = torch.Tensor(y_hot).view(-1, batch_size, input_size)
    # 实际使用 NLLLoss()损失函数的时候不需要进行one-hot编码
    labels = torch.LongTensor(y_data).view(-1, 1)
    print("labels:", labels)

    # 基础模型
    net = Model(input_size, hidden_size, batch_size)
    # loss 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 梯度下降优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(20):
        loss = 0
        optimizer.zero_grad()
        # 初始化隐藏层
        hidden = net.init_hidden()
        print('Predicted string: ', end='')

        # RNNCell
        for input, label in zip(inputs, labels):
            hidden = net(input, hidden)
            # print('hidden:',hidden)
            # print('label:', label)
            # 注意，这里的hidden之后会取最大的预测值与label相乘，所以label在实际使用的时候不需要进行one-hot编码
            loss += criterion(hidden, label)
            _, idx = hidden.max(dim=1)
            # print('idx.item()',idx.item())
            print(idx2char[idx.item()], end='')

        loss.backward()
        optimizer.step()
        print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))



