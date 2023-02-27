#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import operator
import numpy as np
from PIL import Image
import os as os


# 1、数据处理。传入一个文件地址,将每个图片转为一个向量。所有图片组成一个特征向量矩阵
def createDataSet(finame_guo, filename_zhong):
    # os.walk()是一个生成器，返回三个值：根目录，根目录下的目录和文件列表
    folder = os.walk(finame_guo)
    # 将返回值转化成列表
    files = list(folder)
    # print(files)
    # 这里得到的file_list就是给的路径下面的文件名称
    file_list_guo = files[0][2]

    folder = os.walk(filename_zhong)
    files = list(folder)
    file_list_zhong = files[0][2]

    sample_data_list = []
    labels_list = []

    # todo 构建 国 数据矩阵
    for imageName in file_list_guo:
        # 拼接国的地址
        new_fileaddress = finame_guo + '/' + imageName
        group_guo = Image.open(new_fileaddress)
        group_guo = np.copy(group_guo)
        # 中图片矩阵转化为一维向量
        group_guo = group_guo.reshape(1, -1)

        sample_data_list.append(group_guo)
        labels_list.append('国')

    # todo 构建 中 数据矩阵
    for imageName in file_list_zhong:
        new_fileaddress = filename_zhong + '/' + imageName
        group_guo = Image.open(new_fileaddress)
        group_guo = np.copy(group_guo)
        # 国图片数据改变成一维向量
        group_guo = group_guo.reshape(1, -1)

        sample_data_list.append(group_guo)
        labels_list.append('中')

    # 列表转化为数组
    sample_data = np.array(sample_data_list)
    labels = np.array(labels_list)

    return sample_data, labels


# KNN 算法过程
# (1)计算已知类别数据集中的点与当前点的距离
# (2)按照距离递增次序排序，选取与当前点距离最小的 k 个点
# (3)确定前 k 个点所在类别的出现频率
# (4)返回前 k 个点出现频率最高的类别作为当前点的预测分类
def KNN_classify(k, dis, X_train, labels, Y_test):
    assert dis == 'E' or dis == 'M'
    num_test = Y_test.shape[0]
    # 测试样本的数量
    labellist = []
    '''
    使用欧拉公式作为距离变量
    '''
    if (dis == 'E'):
        for i in range(num_test):
            # 实现欧式距离公式
            # np.tile用于扩充数组维数.将 Y_test复制80份 (80, 14700)
            target = np.tile(Y_test[i], (X_train.shape[0], 1))

            X_train = X_train.reshape(-1, 14700)

            distance = (X_train - target) ** 2
            distances = np.sqrt(np.sum(distance, axis=1))  # axis 行求和
            # distances = np.sqrt(np.sum(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2),axis=1))

            # 实现曼哈顿距离公式
            # distances = np.sum(np.abs(X_train-np.tile(Y_test[i],(x_train.shape[0],1))))

            # 距离由小到大进行排序，并返回index值
            nearest_k = np.argsort(distances)

            # 选取前k个距离
            topK = nearest_k[:k]
            print('前k个最接近目标向量的向量地址下标 top k=', k, topK)

            classCount = {}
            # 统计每个类别的个数，{}字典
            for i in topK:
                # .get()函数用于获取指定键的值，get(labels[i], 0),后面的0是初始值
                classCount[labels[i]] = classCount.get(labels[i], 0) + 1

            print("前k个预测地址进行分类统计后的结果", classCount)
            # 每轮字典分类排序
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labellist.append(sortedClassCount[0][0])

        return np.array(labellist)


if __name__ == '__main__':
    # 样本文件目录
    address_guo = 'D:/pycharmworkspace/Datas/data_set/guo'
    address_zhong = 'D:/pycharmworkspace/Datas/data_set/zhong'

    # 构建样本数据和分类标签
    sample_data, labels = createDataSet(address_guo, address_zhong)

    # 测试分类图片
    test_zhong = Image.open('D:/pycharmworkspace/Datas/data_set/24.png')
    test_zhong = np.copy(test_zhong)
    test_zhong = test_zhong.reshape(1, -1)

    # KNN分类 E代表欧式距离，M代表曼哈顿距离
    y_test_pred = KNN_classify(5, 'E', sample_data, labels, test_zhong)

    print('最后的预测结果', y_test_pred)
