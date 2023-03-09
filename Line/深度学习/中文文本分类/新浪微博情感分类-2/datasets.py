#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
import jieba
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SinaWeibo4DataSet(Dataset):
    # torch.utils.datasources.Dataset是代表自定义数据集方法的类，用户可以通过继承该类来自定义自己的数据集类，
    # 在继承时要求用户重载__len__()和__getitem__()这两个魔法方法。
    def __init__(self, voc_dict_path, data_path, data_stop_path):
        # 数据量不大，采用一次性全部直接加载到内存中
        # 字典路径，数据路径，停用词路径
        self.voc_dict_path = voc_dict_path
        self.data_path = data_path
        self.data_stop_path = data_stop_path

        self.voc_dict = read_dict(voc_dict_path)
        self.data, self.max_len_seq = load_data(self.data_path, self.data_stop_path)

        # 打乱data里面数据
        np.random.shuffle(self.data)

    # 实现了能够通过索引的方法获取对象中的任意元素
    # 我们可以在__getitem__()中实现数据预处理。一次处理一个标签，一条处理
    def __getitem__(self, index):
        data_detial = self.data[index]
        label = int(data_detial[0])
        # print(label)
        word_list = data_detial[1]
        # 词在字典的索引下标，不存在的使用"UNK" 的索引，就是最后的排序。最后看看长度是否一样,不一样的话数据进行拓展，拓展到长度和最长的一样
        input_idx = []
        for item in word_list:
            if item in self.voc_dict.keys():
                input_idx.append(self.voc_dict[item])
            else:
                input_idx.append(self.voc_dict["<UNK>"])

        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict["<PAD>"] for _ in range(self.max_len_seq - len(input_idx))]

        # 最后将列表转为数组，为了后面的四则运算，列表不可以进行，数组才可以
        data = np.array(input_idx)
        return label, data

    # 返回的是数据集的大小,
    # 我们构建的数据集是一个对象，而数据集不像序列类型（列表、元组、字符串）那样可以直接用len()来获取序列的长度，
    # 魔法方法__len__()的目的就是方便像序列那样直接获取对象的长度。
    def __len__(self):
        return len(self.data)


# 读取字典
def read_dict(path):
    voc_dict = {}
    dict_list = open(path).readlines()
    for item in dict_list:
        item = item.split(",")
        # 转为整型，不然默认是字符串，报错default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <U3
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict


# 加载数据并做分词处理---标签、分词数据、最大句子长度。通过字典进行编码，不是通过word2Voctor
def load_data(data_path, stop_path):
    # 获取数据并去掉标题,一行为单位的list
    # data_list = open(data_path, encoding="utf-8").readlines()[1:]

    # 测试使用
    data_list = open(data_path).readlines()

    # data_lis = open(data_path, encoding="utf-8").readlines()[1:]
    # data_list = data_lis[1:10]+data_lis[70000:70010]
    print(data_list)

    # 获取停用词数据、过滤停用词中的换行符
    stops_word = open(stop_path, encoding="utf-8").readlines()
    stops_word = [line.strip() for line in stops_word]
    stops_word.append(" ")
    stops_word.append("\ufeff")
    stops_word.append("\n")
    # print(stops_word)

    # 词频---字典
    voc_dict = {}
    # 分词处理后的数据---标签，分词结果
    data = []
    # 句子的最大长度
    max_len_seq = 0
    for item in data_list:
        label = item[0]
        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        content = item[1:].strip()
        seg_list = jieba.cut(content, cut_all=False)

        seg_res = []
        for seg_item in seg_list:
            # 分词去掉停用词
            if seg_item in stops_word:
                continue
            seg_res.append(seg_item)
            # 词典构建
            if seg_item in voc_dict.keys():
                voc_dict[seg_item] = voc_dict[seg_item] + 1
            else:
                voc_dict[seg_item] = 1

        if len(seg_res) > max_len_seq:
            max_len_seq = len(seg_res)
        # 分类标签以及分词结果保持在data里面
        data.append([label, seg_res])

    return data, max_len_seq


# 定义一下data_load的相关设置
def Data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)

#
# if __name__ == '__main__':
#     from config import config
#
#     # 数据路径
#     data_path = "datasources/weibo_senti_100k.csv"
#     # 停用词路径
#     stop_path = "datasources/hit_stopwords.txt"
#     # 字典路径
#     dict_path = "datasources/dict.txt"
#     config = config()
#     dataset = SinaWeibo4DataSet(dict_path, data_path, stop_path)
#     tran_DataLoad = Data_loader(dataset,config)
#     for id, batch in enumerate(tran_DataLoad):
#         print(batch)
