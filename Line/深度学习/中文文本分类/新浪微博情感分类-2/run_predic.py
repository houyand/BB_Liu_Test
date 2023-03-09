#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import Data_loader, SinaWeibo4DataSet
from config import config

# 字典路径
dict_path = "datasources/dict.txt"
# 数据路径
data_path = "datasources/test.txt"
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
model.load_state_dict(torch.load("models/final.pth"))

with torch.no_grad():
    for i, batch in enumerate(tran_DataLoad):
        label, data = batch
        # 此处的data可能还是数组，需要转为tensor,并且送入GPU  后面测试一下data_load是否已经处理了,data 数据要转为long型才能送入embedding层
        data = data.long()
        data = torch.tensor(data).to(config.devices)
        label = torch.tensor(label, dtype=torch.int64).to(config.devices)
        # data.to(config.devices)
        # label.to(config.devices)

        # 前向传播，预测
        predict_softmax = model.forward(data)

        print("predict_softmax is: {}".format(predict_softmax))
        print("label :{}".format(label))

        pred = torch.argmax(predict_softmax, dim=1)
        print("pred is: {}".format(pred))

        # 统计预测的准确率
        out = torch.eq(pred, label)
        print(out)
        print(out.sum() * 1.0 / pred.size()[0])
