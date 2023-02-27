#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import collections
import math
import torch
from torch import nn
from d2l import torch as d21


# d2l是李沐大神自己写的库函数
# 编码器  没有最后一层的GRU
class Seq2SeqEncoder(d21.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, input1, *args):
        input1 = self.embedding(input1)

        # permute 维度置换   原来的tensor torch.Size([1, 2, 3, 4])    转置后的tensor torch.Size([3, 2, 1, 4])
        input1 = input1.permute(1, 0, 2)

        output, state = self.rnn(input1)

        return output, state


# 解码器  GRU
class Seq2SeqDecoder(d21.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)

        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, encoder_outputs, *args):
        return encoder_outputs[1]

    def forward(self, input1, state):
        # permute 改变batch_size 放在第一个位置
        input1 = self.embedding(input1).permute(1, 0, 2)
        # context上下文 state[-1] 就是encoder 最后输出的隐藏层，repeat 重复复制几次达到和解码器输入一样的维度
        context = state[-1].repeat(input1.shape[0], 1, 1)

        input_and_context = torch.cat((input1, context), 2)

        output, state = self.rnn(input_and_context, state)

        output = self.dense(output).permute(1, 0, 2)

        return output, state


def sequence_mask(input, valid_len, value=0):
    """"通过 0 值化 在序列中屏蔽不相关的项"""
    maxlen = input.size(1)
    # 将小于valid_len 的向量地址取出来
    mask = torch.arange((maxlen), dtype=torch.float32, device=input.device)[None, :] < valid_len[:, None]
    # 这里~mask 表示除了这些地址外的地方设置为 0
    input[~mask] = value
    return input


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """通过拓展softmax交叉熵损失函数来屏蔽不相关的预测 就是把不相关的向量设置为零"""

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # mean 每个句子取一个平均loss,这样得到整个样本loss  unweighted_loss * weights注意这里是交叉熵乘法
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


if __name__ == '__main__':
    # 编码器  vocab_size 词典大小
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # 写个eval表示encoder不会生效，不写也可以
    encoder.eval()
    input = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(input)
    print(output.shape)
    print(state.shape)

    # 解码器
    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()
    state1 = decoder.init_state(encoder(input))
    output1, state1 = decoder(input, state1)
    print(output1.shape)
    print(state1.shape)

    # MaskedSoftmaxCELoss 测试例如：
    loss = MaskedSoftmaxCELoss()
    print(loss(torch.ones(3, 4, 10),
               torch.ones((3, 4), dtype=torch.long),
               torch.tensor([4, 2, 0])
               )
          )
    # tensor([2.3026, 1.1513, 0.0000])

    # sequence_mask 效果如下
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(sequence_mask(x, torch.tensor([1, 2])))
    # tensor([[1,0,0],[4,5,0]])
