from math import log
import operator

import pickle

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import lines as mlines

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def createDataSet():
    """
    函数说明:创建测试数据集

    Parameters:
        无
    Returns:
        dataSet - 数据集
        labels - 分类属性
    """
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性


def calcShannonEnt(dataSet):
    '''
    函数说明:计算给定数据集的经验熵(香农熵)
    Parameter:
        dataSet - 数据集
    Returns:
        shannonEnt - 经验熵(香农熵)
    '''
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签(label)出现次数的字典
    labelCounts = {}
    # 特征向量统计
    for featVec in dataSet:
        # 读取当前特征向量标签(label)
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # label 计数
        labelCounts[currentLabel] += 1
    # 经验熵初始化
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算该标签的概率
        prob = float(labelCounts[key]) / numEntires
        # 经验熵计算
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    函数说明：按照给定特征划分数据集
    Parameters:
        dataSet - 待划分的数据集
        axis - 划分数据集的特征
        value - 需要返回的特征的值
    Returns:
        retDataSet - 划分后的数据集
    '''
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    函数说明：选择最优特征
    Parameters:
        dataSet - 数据集
    Returns:
        bestFeature - 信息增益最大的(最优)特征的索引值
    '''
    # 特征数量
    numFeature = len(dataSet[0]) - 1
    # 计算数据集的经验熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益初始化
    bestInfoGain = 0.0
    # 最有特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeature):
        # 获取dataSet的第i个特征
        featList = [example[i] for example in dataSet]
        # 创建set集合，元素不可重复
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # 得到划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print(f'第{i}个特征的增益为{infoGain}')
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majortyCnt(classList):
    '''
    统计classList中出现次数最多的元素(类标签)
    Parameters:
        classList - 类标签列表
    Returns:
        sortedClassCount[0][0] - 出现次数最多的元素(类标签)
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    '''
    函数说明：创建决策树
    :param dataSet: 训练数据集
    :param labels: 分类属性标签
    :param featLabels: 储存选择的最优特征标签
    :return: myTree - 决策树
    '''
    # 取分类标签
    classList = [example[-1] for example in dataSet]
    # 如果类别标签完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完全部标签时返回
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majortyCnt(classList)
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 使用最优标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已使用的特征标签
    del (labels[bestFeat])
    # 得到最优特征中的全部取值
    featValues = [example[bestFeat] for example in dataSet]
    # 去重 得到唯一的可取属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''
    函数说明：使用决策书分类
    :param inputTree:  已经生成的决策树
    :param featLabels:  储存选择的最优特征标签
    :param testVec:  测试数据列表，顺序对应最优特征标签
    :return:  classLabel - 分类结果
    '''
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.keys()
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    '''
    函数说明:储存决策树
    :param inputTree: 已经生成的决策书
    :param filename:  决策树的储存文件命
    :return: None
    '''
    with open(filename, 'wb') as fw:
        pickle.dumps(inputTree, fw)


def grabTree(filename):
    '''
    函数说明：读取决策树
    :param filename: 决策树的储存文件命
    :return: pickle.load(fr) - 决策书字典
    '''
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # print('最优特征索引值：', chooseBestFeatureToSplit(dataSet))
    featLabels = []
    print('决策树为', createTree(dataSet, labels, featLabels))
