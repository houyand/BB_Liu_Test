#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import Data_loader, SinaWeibo4DataSet
from config import config

# 字典路径
dict_path = "datasources/dict.txt"
# 数据路径
data_path = "datasources/weibo_senti_100k.csv"
# 停用词路径
stop_path = "datasources/hit_stopwords.txt"

# 加载数据
dataset = SinaWeibo4DataSet(dict_path, data_path, stop_path)
config = config()
tran_DataLoad = Data_loader(dataset, config)

config.pad_size = dataset.max_len_seq

# 模型
model = Model(config)
# 模型使用GPU
model.to(config.devices)

# 定义损失函数，交叉熵损失
loss = nn.CrossEntropyLoss()

# 定义优化器，学习率为config.learn_rate
optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)

for epoch in range(config.num_epochs):
    for i, batch in enumerate(tran_DataLoad):
        label, data = batch
        # 此处的data可能还是数组，需要转为tensor,并且送入GPU  后面测试一下data_load是否已经处理了,data 数据要转为long型才能送入embedding层
        data = data.long()
        data = torch.tensor(data).to(config.devices)
        label = torch.tensor(label, dtype=torch.int64).to(config.devices)
        # data.to(config.devices)
        # label.to(config.devices)

        # 优化器归零
        optimizer.zero_grad()
        # 前向传播，训练
        predict = model.forward(data)
        # 计算损失
        loss_val = loss(predict, label)

        print("epoch is {} , batch_num is {} , loss is {}".format(epoch, i, loss_val))
        # 反向传播
        loss_val.backward()
        # 参数梯度更新
        optimizer.step()

# 保存模型
torch.save(model.state_dict(), "models/final.pth")
