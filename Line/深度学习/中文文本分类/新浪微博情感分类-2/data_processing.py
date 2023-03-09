#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import jieba

# 数据路径
data_path = "datasources/weibo_senti_100k.csv"
# 停用词路径
stop_path = "datasources/hit_stopwords.txt"

# 获取数据并去掉标题,一行为单位的list
data_list = open(data_path, encoding="utf-8").readlines()[1:]
# print(data_list)

# 获取停用词数据、过滤停用词中的换行符
stops_word = open(stop_path, encoding="utf-8").readlines()
stops_word = [line.strip() for line in stops_word]
stops_word.append(" ")
stops_word.append("\ufeff")
stops_word.append("\n")
# print(stops_word)

# 词典
voc_dict = {}
# 最小词典大于1
min_seq = 1
# 词典取前1000个高频词
top_n = 1000

# 词频小于1的词定义为UNK
UNK = "<UNK>"
# 后面NLP 任务的时候使用
PAD = "<PAD>"

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

    # print(content)
    # print(seg_res)

# 排序取前1000词
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x: x[1],
                  reverse=True)[:top_n]
# print(voc_list)
# 更新voc_dict  出现频率最高的从 0 开始排序  word_count是这种形式(“多大”：12)，idx是从0开始
voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}

# 不是很明白要干啥 ，统计一下字典有多大？
voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})

# 保持训练完的字典数据
ff = open("datasources/dict.txt", "w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))
