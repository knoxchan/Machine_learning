from sklearn import tree
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pydotplus
from io import StringIO

import os

# 附加环境变量的方法
os.environ['path'] += os.pathsep + 'D:\Software\Graphviz 2.44.1\\bin'


# lenses 数据解析
# 年龄 症状 是否散光 眼泪数量 最终分类标签

def file2DataFrame():
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 最终分类标签提取
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lenses_labels:
        for each in lenses:
            lenses_list.append(each[lenses_labels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
    for col in lenses_pd.columns:  # 为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    return lenses_pd, lenses_target


def lensesTree(data, target):
    # 创建分类树 最大深度为4
    clf = tree.DecisionTreeClassifier(max_depth=4)
    # 使用数据 构建分类树
    clf.fit(data.values.tolist(), target)

    # 使用graphviz 显示树状图
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(), class_names=clf.classes_, filled=True,
                         rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')


if __name__ == '__main__':
    lenses_pd, lenses_target = file2DataFrame()
    lensesTree(lenses_pd, lenses_target)
