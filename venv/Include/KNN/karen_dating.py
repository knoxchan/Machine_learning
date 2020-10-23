import numpy as np
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import lines as mlines

import operator

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

'''
1.收集数据：提供文本文件
2.准备数据：使用pyhton解析文本文件   file2matrix()
3.分析数据：使用matplotlib画二位扩散图
4.训练算法：knn不需要进行训练
5.测试算法：使用karen提供的部分数据作为测试样本
6.使用算法：之后karen可以输入一些特征数据判断对方是否是自己喜欢的类型
'''


def knn_main():
    dating_data_mat, dating_labels = file2matrix('karen_dating.txt')
    showdatas(dating_data_mat,dating_labels)

def file2matrix(filename):
    # karen_dating 特征
    # 每年获取的飞行里程数 玩视频游戏所耗时间百分比 每周消费的冰淇凌公升数
    # 结果 ： 不喜欢 魅力一般 极具魅力
    fr = open(filename)
    array_lines = fr.readlines()
    num_of_lines = len(array_lines)
    # 特征矩阵 定义
    return_mat = np.zeros((num_of_lines, 3))
    # 结果向量 定义
    class_label_vector = []
    index = 0
    for line in array_lines:
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        if list_from_line[-1].strip() == 'didntLike':
            class_label_vector.append(1)
        elif list_from_line[-1].strip() == 'smallDoses':
            class_label_vector.append(2)
        else:
            class_label_vector.append(3)
        index += 1
    return return_mat, class_label_vector


def showdatas(datingDataMat, datingLabels):
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
    axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)09 876542第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
    axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
    axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
    axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
    axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

def classify0(inX, dataSet, labels, k):
	#numpy函数shape[0]返回dataSet的行数
	dataSetSize = dataSet.shape[0]
	#在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	#二维特征相减后平方
	sqDiffMat = diffMat**2
	#sum()所有元素相加,sum(0)列相加,sum(1)行相加
	sqDistances = sqDiffMat.sum(axis=1)
	#开方,计算出距离
	distances = sqDistances**0.5
	#返回distances中元素从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
	#定一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		#计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	print(sortedClassCount)
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

def autoNorm(dataSet):
	#获得数据的最小值
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	#最大值和最小值的范围
	ranges = maxVals - minVals
	#shape(dataSet)返回dataSet的矩阵行列数
	normDataSet = np.zeros(np.shape(dataSet))
	#返回dataSet的行数
	m = dataSet.shape[0]
	#原始值减去最小值
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	#除以最大和最小值的差,得到归一化数据
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	#返回归一化数据结果,数据范围,最小值
	return normDataSet, ranges, minVals

def datingClassTest():
	#打开的文件名
	filename = "karen_dating.txt"
	#将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
	datingDataMat, datingLabels = file2matrix(filename)
	#取所有数据的百分之十
	hoRatio = 0.10
	#数据归一化,返回归一化后的矩阵,数据范围,数据最小值
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#获得normMat的行数
	m = normMat.shape[0]
	#百分之十的测试数据的个数
	numTestVecs = int(m * hoRatio)
	#分类错误计数
	errorCount = 0.0

	for i in range(numTestVecs):
		#前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
			datingLabels[numTestVecs:m], 4)
		print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))


if __name__ == '__main__':
    datingClassTest()
