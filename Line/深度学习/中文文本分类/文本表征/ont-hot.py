#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
独热编码
1、手动实现
2、使用Sklearn 包
"""
import jieba
import numpy as np
from sklearn.preprocessing import LabelBinarizer


# 1、手动实现
def one_hot_hand(corpus):
    words = []
    # 分词
    for corpu in corpus:
        words.extend(corpu.split())

    # 去重
    words = list(set(words))
    # 构建词表
    word_dict = {}
    for i, word in enumerate(words):
        word_dict[word] = i

    # # 基于词表进行编码
    for corpu in corpus:
        word1 = corpu.split()
        # 文字转为下标
        word1_index = [word_dict[_] for _ in word1]
        # 下标变为one-hot,得到总的one-hot编码
        sum_hot = [get_ong_hot(i, len(word_dict)) for i in word1_index]

        sum_hot = np.array(sum_hot)

        return sum_hot


def get_ong_hot(index, length):
    """
    获取一个ong-hot编码
    """
    ont_hot = [0 for i in range(length)]
    # 指定位置为 1
    ont_hot[index] = 1

    return np.array(ont_hot)


# 2、使用Sklearn 包编码
def use_Sklearn_encoder(corpus):
    words = []
    # 分词
    for corpu in corpus:
        words.extend(corpu.split())
    # 去重后所有的词
    words = list(set(words))
    # 初始化，就是计算好了向量的长度，与词表的长度一样
    lb = LabelBinarizer()
    lb.fit(words)

    sum_hot = []
    for corpu in corpus:
        # 取句子并分词
        sentence = corpu.split()
        hot = lb.transform(sentence)
        sum_hot.append(hot)
    sum_hot = np.array(sum_hot)
    return sum_hot


# 使用Sklearn 包解码
def use_Sklearn_decoder(corpus, sum_hot):
    words = []
    # 分词
    for corpu in corpus:
        words.extend(corpu.split())
    # 去重后所有的词
    words = list(set(words))
    # 初始化，就是计算好了向量的长度，与词表的长度一样
    lb = LabelBinarizer()
    lb.fit(words)
    sentence = []
    for hot in sum_hot:
        once_sentence = lb.inverse_transform(hot)
        sentence.append(once_sentence)

    return sentence


if __name__ == "__main__":
    # 语料
    corpus = ["这 是 一个 文档", "这 是 第二 个 文档", "这 是 第三 个 文档", "这 是 第四 个文档"]

    # 1、手动编码
    # put = one_hot_hand(corpus)
    # print(put)

    # 2、使用Sklearn,一次一个句子
    # 编码
    put = use_Sklearn_encoder(corpus)
    print(put)
    # 解码
    put = use_Sklearn_decoder(corpus, put)
    print(put)
