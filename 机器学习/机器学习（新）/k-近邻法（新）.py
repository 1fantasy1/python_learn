import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 使用 NumPy 实现 L_p 距离
def L(x, y, p=2):
    """
    计算向量之间的 L_p 距离，默认为欧几里得距离。
    """
    return np.linalg.norm(np.array(x) - np.array(y), ord=p)

# 测试例子
x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

for i in range(1, 5):
    distances = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    print(min(zip(distances.values(), distances.keys())))

#################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 加载数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# 数据可视化
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('花萼长度')
plt.ylabel('萼片宽度')
plt.legend()
plt.show()

#################################################################################################

from collections import Counter

# KNN 分类器
class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        # 计算所有训练点到测试点的距离
        distances = [np.linalg.norm(X - x_train, ord=self.p) for x_train in self.X_train]
        # 选择最近的 n 个点
        nearest_indices = np.argsort(distances)[:self.n]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        # 返回出现频率最高的类别
        return Counter(nearest_labels).most_common(1)[0][0]

    def score(self, X_test, y_test):
        correct_count = sum(self.predict(X) == y for X, y in zip(X_test, y_test))
        return correct_count / len(X_test)

# 测试自定义KNN分类器
from sklearn.model_selection import train_test_split
X, y = df.iloc[:100, [0, 1, -1]].values[:, :-1], df.iloc[:100, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = KNN(X_train, y_train)
print("自定义KNN分类器的准确率：", clf.score(X_test, y_test))

#################################################################################################

# KD 树节点定义
class KdNode:
    def __init__(self, dom_elt, split, left=None, right=None):
        self.dom_elt = dom_elt  # k维向量节点
        self.split = split  # 切分维度
        self.left = left  # 左子树
        self.right = right  # 右子树

# KD 树的构建
class KdTree:
    def __init__(self, data):
        def CreateNode(split, data_set):
            if not data_set:
                return None
            data_set.sort(key=lambda x: x[split])
            mid = len(data_set) // 2
            next_split = (split + 1) % len(data_set[0])
            return KdNode(data_set[mid], split,
                          CreateNode(next_split, data_set[:mid]),
                          CreateNode(next_split, data_set[mid + 1:]))
        self.root = CreateNode(0, data)

# 搜索最近邻
from collections import namedtuple
result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")

def find_nearest(tree, point):
    k = len(point)

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)

        nodes_visited = 1
        s = kd_node.split
        pivot = kd_node.dom_elt

        if target[s] <= pivot[s]:
            nearer_node, further_node = kd_node.left, kd_node.right
        else:
            nearer_node, further_node = kd_node.right, kd_node.left

        nearest = travel(nearer_node, target, max_dist)
        best = nearest.nearest_point
        dist = nearest.nearest_dist
        nodes_visited += nearest.nodes_visited

        if dist < max_dist:
            max_dist = dist

        axis_dist = abs(pivot[s] - target[s])
        if max_dist < axis_dist:
            return result(best, dist, nodes_visited)

        pivot_dist = np.linalg.norm(np.array(pivot) - np.array(target))
        if pivot_dist < dist:
            best, dist = pivot, pivot_dist
            max_dist = dist

        further_nearest = travel(further_node, target, max_dist)
        nodes_visited += further_nearest.nodes_visited
        if further_nearest.nearest_dist < dist:
            best, dist = further_nearest.nearest_point, further_nearest.nearest_dist

        return result(best, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))

# 构建KD树并搜索最近邻
data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KdTree(data)
point_test = [3,4.5] # 输入测试点
ret = find_nearest(kd, point_test)
print(f"点{point_test}的最近邻点为：{ret.nearest_point},距离为：{ret.nearest_dist},访问的节点数为：{ret.nodes_visited}")