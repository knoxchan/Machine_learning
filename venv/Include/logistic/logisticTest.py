import matplotlib.pyplot as plt
import numpy as np


def loadData():
    '''
    函数说明：加载数据
    :return:
        dataMat - 数据列表
        labelMat - 标签列表
    '''
    dataMat = []
    labelMat = []
    fr = open('./testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


def plotDataSet():
    '''
    函数说明：绘制数据集
    :return:
    '''
    dataMat, labelMat = loadData()
    dataArr = np.array(dataMat)
    # 数据个数
    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=0.5)
    ax.scatter(xcord2, ycord2, s=20, c='g', alpha=0.5)
    plt.title('dataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['+', '-'])
    weights = gradscent(dataMat,labelMat)
    x = np.arange(-4,3,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.show()



def sigmoid(inx):
    '''
    函数说明 sigmoid 函数
    :param inx: 数据
    :return:
        sigmoid 函数
    '''
    return 1.0 / (1 + np.exp(-inx))


def gradscent(dataMatIn, classLabels):
    '''
    函数说明： 梯度上升算法
    :param dataMatIn:  数据集
    :param classLabels:  数据标签
    :return:
        weights.getA() - 求得的权重数组(最优参数)
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights += alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    函数说明 ： 随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    '''
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化                                        #存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))           #选择随机选取的一个样本，计算h
            error = classLabels[dataIndex[randIndex]] - h                            #计算误差
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]   #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights

if __name__ == '__main__':
    plotDataSet()
