#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

from torch.utils.data import Dataset, DataLoader
import gzip
import csv

# 1、准备参数
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 10
N_CHARS = 128
USE_GPU = False


# 基础模型
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # 是否是双向RGU,如果是2表示双向，hidden将是两个方向隐藏层拼接出来 ，
        # 所以注意看 self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)  hidden_size * self.n_directions
        self.n_directions = 2 if bidirectional else 1
        # 嵌入层
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # GRU输入为hidden_size，输出hidden_size
        self.gru = torch.nn.GRU(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=n_layers,
                                bidirectional=bidirectional)

        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):

        # 数据处理 input shape : B x S -> S x B   为embedding 做准备，矩阵转置
        input = input.t()

        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)

        # inputs是torch.Size(seqLen, batch, input-size)
        # embedding是torch.Size(batch, seqLen, C)
        embedding = self.embedding(input)
        # pack them up 将embedding 后的数据再次进行打包为一个GRU,LSTM 能处理的data。
        # seq_lengths是每个单词的长度 pack_padded_sequence处理后用batch_sizes这个变量记录seq_lengths
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)

        if self.n_directions == 2:
            # 循环GRU 拼接输出的隐藏层
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)
        return fc_output


# 处理名字国家数据
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'D:/学习资料/研究生/人工智能资料/PyTorch深度学习实践/names_train.csv/names_train.csv' if is_train_set else 'D:/学习资料/研究生/人工智能资料/PyTorch深度学习实践/names_test.csv/names_test.csv'
        with open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.names = [row[0] for row in rows]
            self.len = len(self.names)
            self.countries = [row[1] for row in rows]
            self.country_list = list(sorted(set(self.countries)))  # set 去重，sort排序
            self.country_dict = self.getCountryDict()  # 构建国家词典
            self.country_num = len(self.country_list)

    # self.names[index]获取到姓名，self.countries[index]取到国家名字 self.country_dict[self.countries[index]]国家的词典编号
    def __getitem__(self, index):
        country_name = self.countries[index]
        country_id = self.country_dict[country_name]

        return self.names[index], country_id

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


# 定义一些方法
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 姓名每个字符转化为 ASSCII 码
def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)

    return tensor


# 准备name数据
def make_tensors(names, countries):
    # 所有名字的对应asscii编码矩阵与每个名字长度集合
    sequences_and_lengths = [name2list(name) for name in names]
    # 拿到编码矩阵
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    # 序列长度转化为一个LongTensor
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name, BatchSize x SeqLen
    # 名字编码矩阵 做padding 0处理，不够长的加0 这样就是同样长度了，
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    # seq_lengths是排序后的，perm_idx 是排序后的下标
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # 使用perm_idx更新seq_tensor 排序
    seq_tensor = seq_tensor[perm_idx]
    # 使用perm_idx更新countries 排序
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')

    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':

    # 2、准备data
    trainset = NameDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = NameDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    # 总的国家类别数量，也就是模型输出的维度大小
    N_COUNTRY = trainset.getCountriesNum()

    # 3、准备模型
    # N_CHARS（input_size） 每个字符变为一个独热向量，字母表有多少个独热向量，HIDDEN_SIZE GRU输出的隐层大小，N_COUNTRY（output_size） 分类的国家有多少，N_LAYER GRU层数
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    # 是否使用GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 4、开始训练
    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    # 5、打印训练结果参数
    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
