# 案例 疝气病症状预测病马的死亡率
'''
这里的数据包含了368个样本和28个特征。这种病不一定源自马的肠胃问题，其他问题也可能引发马疝病。
该数据集中包含了医院检测马疝病的一些指标，有的指标比较主观，有的指标难以测量，例如马的疼痛级别。
另外需要说明的是，除了部分指标主观和难以测量外，该数据还存在一个问题，数据集中有30%的值是缺失的。
下面将首先介绍如何处理数据集中的数据缺失问题，然后再利用Logistic回归和随机梯度上升算法来预测病马的生死
'''

'''
数据预处理:
    - 如果测试集中数据的特征标签已经确定 使用0来进行替代(sigmoid(0) = 0.5)不会对结果产生影响
    - 如果测试集中数据的特征标签缺失 直接删除该条数据
'''

import numpy as np
import random


def sigmoid(inx):
    '''
    函数说明 - sigmoid函数
    :param inx: 数据
    :return: sigmoid处理后的数据
    '''
    return 1.0 / (1 + np.exp(-inx))


def stocGrandAscent1(dataMatrix, classLabels, numIter=150):
    '''
    函数说明: 改进的随机梯度上升算法 Stochastic Gradient Ascent
    :param dataMatrix: 数据数组
    :param classLabels: 数据标签数组
    :param numIter: 迭代次数
    :return: weights - 求得的最优回归系数数组
    '''
    # 返回数据数组的大小 m行 n列
    m, n = np.shape(dataMatrix)
    # 创建权重数组 长度为特征数量
    weights = np.ones(n)
    # 重复训练numIter次数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次循环降低alpha大小  每次减小 1/(i+j)
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 随机选择一个样本 计算h
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            # 更新回归参数
            weights += alpha * error * dataMatrix[dataIndex[randIndex]]
            del (dataIndex[randIndex])
    return weights


def grandAscent(dataMatrix, classLabels):
    '''
    函数说明：梯度上升算法
    :param dataMatrix: - 数据集
    :param classLabels: - 数据标签
    :return:
        weights 求得的最优权重数组
    '''
    dataMatrix = np.mat(dataMatrix)
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小 m行数 n列数
    m, n = np.shape(dataMatrix)
    alpha = 0.01  # 学习率
    maxCycle = 500  # 迭代次数
    weights = np.ones((n,1))  # 权重系数初始化
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights += alpha * dataMatrix.transpose() * error
    return weights


def colicTest():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainSet = []
    trainLabels = []
    for line in frTrain.readlines():
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(len(currline) - 1):
            lineArr.append(float(currline[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(currline[-1]))
    # 普通梯度上升算法
    trainWeights = grandAscent(np.array(trainSet), trainLabels)
    # 普通梯度错误次数初始化
    errorCount = 0
    # 随机梯度上升算法
    stocTrainWeights = stocGrandAscent1(np.array(trainSet), trainLabels,300)
    # 随机梯度错误次数初始化
    stocErrorCount = 0
    # 测试总次数初始化
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(len(currline) - 1):
            lineArr.append(float(currline[i]))
        if int(classifyVector(np.array(lineArr),trainWeights[:,0])) != int(currline[-1]):
            errorCount += 1
        if int(classifyVector(np.array(lineArr),stocTrainWeights)) != int(currline[-1]):
            stocErrorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100
    stocErrorRate = (float(stocErrorCount)/numTestVec) * 100
    print(f'普通梯度算法错误率为{errorRate}')
    print(f'随机梯度算法错误率为{stocErrorRate}')

def classifyVector(inX, weights):
    '''
    函数说明：分类函数
    :param inX:
    :param weights:
    :return:
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

if __name__ == '__main__':
    colicTest()
