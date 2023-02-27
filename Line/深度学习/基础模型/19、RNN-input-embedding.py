#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)  # input 进行embedding 编码  4维变为10维
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)   #注意要求batch_first 则inputs batch_first放在第一位
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)  # (num_layers, batch_size, hidden_size)
        # inputs编码
        x = self.emb(x)
        print('inputs-embedding:', x)   #说白了就是改变了编码方式   inputs是torch.Size([5, 1, 4])(seqLen, batch, input-size) 现在是torch.Size([1, 5, 10]) (batch, seqLen, embeddingSize)
        print('inputs-shape:', x.shape)
        x, hidden = self.rnn(x, hidden)
        print('xxxxxx',hidden)  #_hidden torch.Size([2, 1, 8]) (num_layers, batch_size, hidden_size)
        print('inputs-shape:', x)  #inputs-shape  embeddingSize=>hidden_size: torch.Size([1, 5, 8])  (batch, seqLen, hidden_size)
        x = self.fc(x)
        return x.view(-1, num_class)


if __name__ == '__main__':
    # parameters
    # 分类数量
    num_class = 4
    input_size = 4
    hidden_size = 8
    embedding_size = 10
    num_layers = 2
    batch_size = 1
    seq_len = 5

    # 准备输入数据，准备输出结果。多分类任务
    idx2char = ['e', 'h', 'l', 'o']  # 词编码0-e,1-h,2-l,3-o
    # x_data = [1, 0, 2, 2, 3]  # input:     Hello
    x_data = [[1, 0, 2, 2, 3]]  #词嵌入只需要把数值下标统计为列表就可以了，代表一个one-hot 第x位为1，其他的都是零
    y_data = [3, 1, 2, 3, 2]  # output:    ohlol

    inputs = torch.LongTensor(x_data)
    print("inputs:", inputs.size())
    # labels = torch.LongTensor(y_data).view(-1,1)   #多个目标张量([[3],[1],[2],[3],[2]])
    labels = torch.LongTensor(y_data)  # 一个目标张量([3, 1, 2, 3, 2])
    print("labels:", labels)

    # 基础模型
    net = Model()
    # loss 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 梯度下降优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
