#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch

if __name__ == '__main__':
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2  #我这里的疑问是hidden_size与input_size的数值有什么关系
    num_layers = 1  #网络的层数

    # cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers)
    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
    # (seqLen, batchSize, inputSize)
    # inputs = torch.randn(seq_len, batch_size, input_size)
    inputs = torch.randn(batch_size,seq_len, input_size)

    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    out, hidden = cell(inputs, hidden)

    print('Output size:', out.shape)
    print('Output:', out)
    print('Hidden size: ', hidden.shape)
    print('Hidden: ', hidden)

