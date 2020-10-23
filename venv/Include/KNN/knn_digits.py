import numpy as np
from os import listdir
import operator
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(filename):
    '''
    函数说明： 将32*32的二进制图像转换为1*1024向量
    parameters:
        filename - 文件名
    Rreturns:
        return_vect - 返回的而精致图像的1*1024向量
    '''
    # 创建1*1024零向量
    return_vect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行的数据
        line_str = fr.readline()
        # 每一行的32个元素依次添加到return_vect中
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect

def handwriting_class_test():
    '''
    函数说明：手写数字分类测试
    Paramters:
        无
    Returns:
        无
    '''
    # 测试集的labels
    hw_labels = []
    # 返回trainingDigits目录下的文件名
    training_file_list = listdir('trainingDigits')
    # 范围文件夹下的文件个数
    m = len(training_file_list)
    # 初始化训练的mat矩阵，测试集
    training_mat = np.zeros((m, 1024))
    # 从文件名总解析出训练姐的类别
    for i in range(m):
        # 获得文件的名字
        file_name_str = training_file_list[i]
        # 分割过得分类的数字
        class_num = int(file_name_str.split('_')[0])
        # 将获得的类别添加到hw_labels矩阵中
        hw_labels.append(class_num)
        # 将每一个文件的*1024数据储存到training_mat中
        training_mat[i, :] = img2vector(f'trainingDigits/{file_name_str}')

    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型 training_mat 为训练矩阵 hw_labels为对应的标签
    neigh.fit(training_mat, hw_labels)

    # 返回testDigits目录下的文件列表
    test_file_list = listdir('testDigits')
    # 错误检测技术
    error_counts = 0
    # 测试数据的数量
    m_test = len(test_file_list)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(m_test):
        # 获得文件的名字
        file_name_str = test_file_list[i]
        # 获得分类的名字
        class_num = int(file_name_str.split('_')[0])
        # 获取测试集的1*1024向量 用于预测
        testing_mat = img2vector(f'testDigits/{file_name_str}')
        # 取得预测结果
        classifier_result = neigh.predict(testing_mat)

        if classifier_result != class_num:
            error_counts += 1
    print(f'总共错了{error_counts}个数据，错误率是{error_counts / m_test * 100}%')


if __name__ == '__main__':
    handwriting_class_test()
