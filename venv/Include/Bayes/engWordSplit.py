import numpy as np

def loadDataSet():
    '''
    函数说明：创建实验样本
    :return: postingList - 实验样本切分的词条
             classVec - 类别标签向量  0：非侮辱类 1：侮辱类
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec

def createVocabList(dataSet):
    '''
    函数说明： 创建词汇表(文档向量化的第一步)
    :param dataSet:
    :return:vocabSet - 不重复的词条列表 (词汇表)
    '''
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


def setOfWordsVec(vocabList, inputSet):
    '''
    函数说明：根据vocabList词汇表，将inputSet向量化，向量中每个元素为1或0
    :param vocabList: - createVocabList返回的列表
    :param inputList: - 切分的词条列表
    :return: - 文档向量(词条模型)
    '''
    returnVec = [0] * len(vocabList)
    # 遍历输入词表
    for word in inputSet:
        # 检测词条是否存在
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f'the word {word} is not in vocabulary')
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    函数说明：朴素贝叶斯分类器训练函数
    :param trainMatrix:  训练文档矩阵,即setOfWordsVec 返回的returnVec构成的矩阵
    :param trainCategory: 训练类别标签向量，即loadDataSet返回的classVec
    :return:
        p0Vect - 非侮辱类的条件概率数组
        p1Vect - 侮辱类的条件概率数组
        pAbusive - 文档属于侮辱类的概率
    '''
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.zeros(numWords);
    p1Num = np.zeros(numWords)  # 创建numpy.zeros数组,
    p0Denom = 0.0;
    p1Denom = 0.0  # 分母初始化为0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)...
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # 相除
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    函数说明：朴素贝叶斯分类器分类函数
    :param vec2Classify: - 待分类的词条数组
    :param p0Vec: - 侮辱类的条件概率数组
    :param p1Vec: - 非侮辱类的条件概率数组
    :param pClass1: - 文档属于侮辱类的概率
    :return:
        0 - 文档属于非侮辱类
        1 - 文档属于侮辱类
    '''
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWordsVec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation','asd']  # 测试样本1
    thisDoc = np.array(setOfWordsVec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(setOfWordsVec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果

if __name__ == '__main__':
    testingNB()
