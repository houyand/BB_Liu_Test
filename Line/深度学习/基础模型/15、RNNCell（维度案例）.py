#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch

if __name__ == '__main__':
    batch_size = 1
    seq_len = 3  # 向量个数
    input_size = 4  # 线性层，一维向量里面元素个数
    hidden_size = 4

    RNNCell = torch.nn.RNNCell(input_size=input_size,
                               hidden_size=hidden_size)  # 在RNNCell里面输入数据会进行线性变换参数W (input_size*input_size)矩阵

    dataset = torch.randn(seq_len, batch_size, input_size)

    print('dataset',dataset)

    hidden = torch.zeros(batch_size, hidden_size)
    print('hidden first', hidden)
    # 使用RNNCell 需要手动写这个循环，如果是使用的RNN则不需要
    for idx, input in enumerate(dataset):
        print(print('=' * 20, idx, '=' * 20))
        print('Input : ', input)
        print('Input size: ', input.shape)

        hidden = RNNCell(input, hidden)

        print('outputs size: ', hidden.shape)
        print(hidden)
