# coding:utf-8
import operator
import os

import numpy as np
from PIL import Image

'''
Python图像处理库PIL中有九种不同模式。
分别为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F。
模式“1”为二值图像，非黑即白。但是它每个像素用8个bit表示，0表示黑，255表示白
模式“L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
模式“P”为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的。
'''


# 训练集处理，将图片文本转换为数字文本
def fun_pd(src, dst):  # 图片转数值，src源文件，dst目标文件
    if not os.path.exists(src):
        return
    if not os.path.exists(dst):
        os.mkdir(dst)
    list = os.listdir(src)
    # print(list)
    length = len(list)
    for i in range(length):
        path = src + '/' + list[i]
        # SavePath=dst+'/'+list[i][:-10]+"guo"+".txt"
        SavePath = dst + '/' + list[i][:-10] + "zhong" + ".txt"
        read = Image.open(path).convert("1")  # 转化为1模式---非黑（0）即白（255）
        arr = np.asarray(read)
        np.savetxt(SavePath, arr, fmt="%d", delimiter='')  # 保存格式为整数,没有间隔
        np.savetxt(SavePath, arr, fmt="%d")


# src="D:/vs code/hh-code/Python/data_set/zhong"
# dst="D:/vs code/hh-code/Python/data_set/zhong-digit"
# fun_pd(src,dst)
# coding:utf-8

# 获取训练样本和对应的标签
def train_labels(src):
    if not os.path.exists(src):
        return
    list = os.listdir(src)  # os.listdir()输出文件夹中的文件名
    length = len(list)
    labels = []
    train = []
    for i in range(length):
        path = src + '/' + list[i]
        read = open(path)
        temp = []
        for j in range(70):  # 将70*70的矩阵拉成1*4900的矩阵
            line = read.readline()
            lines = line.replace(" ", "")
            for k in range(70):
                bit = int(lines[k])
                temp.append(bit)
        train.append(temp)
        labels.append(path.split('_')[2].split('.')[0])
    train = np.array(train)
    # trains=train.astype(np.int4)
    return train, labels


# 识别
def Classifier(train, label, testpath, KK):
    list = os.listdir(testpath)
    length = len(list)
    errorCount = 0
    for i in range(length):
        # 数据处理
        path = testpath + '/' + list[i]
        # 实际值
        real = path.split('_')[2].split('.')[0]
        read = open(path)
        test = []
        for j in range(70):
            line = read.readline()
            lines = line.replace(" ", "")
            for k in range(70):
                bit = int(lines[k])
                test.append(bit)
        # 计算欧氏距离,不需要遍历，技巧
        m = train.shape[0]
        test = np.array(test)
        # test=test.astype(np.int4)
        test = np.tile(test, (m, 1))

        sum = train - test  # 对应相减
        sum = sum ** 2  # 平方
        sum = np.sum(sum, axis=1)  # 行求和
        sum = sum ** 0.5  # 开方
        # 排序,返回下标
        sum = np.argsort(sum)
        # 前k个，取最大类
        ans = {}
        for n in range(KK):  # 前k个相似点 分类计数
            lab = label[sum[n]]  # 下标对应的标签
            if lab in ans.keys():
                ans[lab] = ans[lab] + 1
            else:
                ans[lab] = 1
        ans = sorted(ans.items(), key=operator.itemgetter(1), reverse=True)  # 降序排列
        print("实际值=", real, "预测值=", ans[0][0])
        if real != ans[0][0]:
            errorCount += 1.0
    print("错误总数：%d" % errorCount)
    print("错误率：%f" % (errorCount / length))


trainpath = "D:/pycharmworkspace/Datas/data_set/train"
testpath = "D:/pycharmworkspace/Datas/data_set/test"

train, label = train_labels(trainpath)
# print(train,label)
Classifier(train, label, testpath, 1)
