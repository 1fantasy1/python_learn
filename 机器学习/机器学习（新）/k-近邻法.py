import math
import time
import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


# 计算 L_p 距离的函数，默认为欧几里得距离（p=2）
def L(x, y, p=2):
    """
    L_p距离公式的实现。可以用来计算曼哈顿距离、欧几里得距离等不同的p范数。

    参数：
    x: 向量 x
    y: 向量 y
    p: 距离范数参数，默认为2（欧几里得距离）

    返回：
    两点之间的距离
    """
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0


# 测试例子：计算 x1、x2 和 x3 之间的不同 p 值下的距离
x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

for i in range(1, 5):
    r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    print(min(zip(r.values(), r.keys())))

# 加载鸢尾花数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# 数据可视化：绘制前50个数据点（类别0）和50到100个数据点（类别1）
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

# 将前100行的数据提取出来用于二分类任务（分类0和1）
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 实现KNN算法
class KNN:
    """
    KNN 分类器的实现

    参数：
    X_train: 训练集特征矩阵
    y_train: 训练集标签向量
    n_neighbors: 临近点的个数，默认为3
    p: 距离度量的类型，默认为欧几里得距离（p=2）
    """

    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        """
        给定一个测试点，预测其类别

        参数：
        X: 测试点特征向量

        返回：
        预测的类别标签
        """
        # 保存最近的n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        # 遍历剩下的点，更新最近的n个点
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        # 统计n个最近邻中各类别的出现次数
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs, key=lambda x: x)[-1]
        return max_count

    def score(self, X_test, y_test):
        """
        计算分类器的准确率

        参数：
        X_test: 测试集特征矩阵
        y_test: 测试集标签向量

        返回：
        分类器的准确率
        """
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


# 测试自定义的KNN分类器
clf = KNN(X_train, y_train)
print("自定义KNN分类器的准确率：", clf.score(X_test, y_test))

# 使用sklearn自带的KNeighborsClassifier进行比较
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
print("sklearn KNN分类器的准确率：", clf_sk.score(X_test, y_test))


# 实现KD-树
class KdNode:
    """
    KD树节点的定义
    """

    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点
        self.split = split  # 切分维度
        self.left = left  # 左子树
        self.right = right  # 右子树


class KdTree:
    """
    KD树的构建
    """

    def __init__(self, data):
        k = len(data[0])  # 数据的维度

        def CreateNode(split, data_set):
            if not data_set:
                return None
            data_set.sort(key=lambda x: x[split])  # 按第split维进行排序
            split_pos = len(data_set) // 2
            median = data_set[split_pos]  # 选取中位数作为当前节点
            split_next = (split + 1) % k  # 计算下一个分割维度
            return KdNode(median, split, CreateNode(split_next, data_set[:split_pos]),
                          CreateNode(split_next, data_set[split_pos + 1:]))

        self.root = CreateNode(0, data)


# 前序遍历KD树
def preorder(root):
    """
    前序遍历KD树
    """
    print(root.dom_elt)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


# KD树最近邻搜索
from math import sqrt
from collections import namedtuple

# 定义一个namedtuple，分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")


def find_nearest(tree, point):
    """
    在KD树中找到与目标点最近的点

    参数：
    tree: KD树对象
    point: 目标点

    返回：
    最近点、距离、访问的节点数
    """
    k = len(point)  # 维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)

        nodes_visited = 1
        s = kd_node.split
        pivot = kd_node.dom_elt

        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)
        nearest = temp1.nearest_point
        dist = temp1.nearest_dist
        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist

        temp_dist = abs(pivot[s] - target[s])
        if max_dist < temp_dist:
            return result(nearest, dist, nodes_visited)

        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))
        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        temp2 = travel(further_node, target, max_dist)
        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))


# 构建 KD 树
data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KdTree(data)
preorder(kd.root)

# 搜索最近邻
ret = find_nearest(kd, [3, 4.5])
print("最近邻点：", ret.nearest_point, "距离：", ret.nearest_dist, "访问的节点数：", ret.nodes_visited)