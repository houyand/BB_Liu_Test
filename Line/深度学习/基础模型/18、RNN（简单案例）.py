
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        # 初始化隐藏层
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


if __name__ == '__main__':
    input_size = 4
    hidden_size = 4
    batch_size = 1
    seq_size = 5
    num_layers = 1

    # 准备输入数据，准备输出结果。多分类任务
    idx2char = ['e', 'h', 'l', 'o']  # 词编码0-e,1-h,2-l,3-o
    x_data = [1, 0, 2, 2, 3]  # input:     Hello
    y_data = [3, 1, 2, 3, 2]  # output:    ohlol

    # 词向量模型[1, 0, 0, 0] 表示e,[0, 1, 0, 0]表示h
    one_hot_lookup = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # 构建输入的词向量
    x_one_hot = [one_hot_lookup[x] for x in x_data]
    print('x_data词向量：', x_one_hot)
    inputs = torch.Tensor(x_one_hot).view(seq_size, batch_size, input_size)

    print("inputs:", inputs)
    print("inputs-shape:", inputs.shape)

    # 实际使用 NLLLoss()损失函数的时候不需要进行one-hot编码
    # labels = torch.LongTensor(y_data).view(-1,1)   #多个目标张量([[3],[1],[2],[3],[2]])
    labels = torch.LongTensor(y_data)                #一个目标张量([3, 1, 2, 3, 2])
    print("labels:", labels)

    # 基础模型
    net = Model(input_size, hidden_size, batch_size, num_layers)
    # loss 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 梯度下降优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


