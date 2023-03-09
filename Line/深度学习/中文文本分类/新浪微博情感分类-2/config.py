#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 模型参数配置类
import torch


class config():
    def __init__(self):
        '''
        self.embedding = nn.Embedding(config.n_vocab,
                                      config.embed_size,
                                      padding_idx=config.n_vocab - 1)

        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)

        self.maxpool = nn.MaxPool1d(config.pad_size)

        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size, config.num_classes)

        self.softmax = nn.Softmax(dim=1)
        '''
        # 字典长度
        self.n_vocab = 1002

        self.embed_size = 128
        # 隐藏层节点数
        self.hidden_size = 256

        # 模型层数
        self.num_layers = 5

        self.dropout = 0.8
        # 二分类问题
        self.num_classes = 2

        # 数据拓展大小，保持与最长句子一致
        self.pad_size = 32

        # 判定是否使用GPU
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 使用dataload 一次取多少条数据
        self.batch_size = 64

        # 配置dataload 取进行下一轮取数据是否打乱数据
        self.is_shuffle = True

        # 学习率
        self.learn_rate = 0.001

        # 训练的轮数
        self.num_epochs = 2
