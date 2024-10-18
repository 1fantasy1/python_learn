import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']

def create_data():
    # 加载数据集
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:150, :])
    # 只选前两列特征用于可视化
    return data[:, :2], data[:, -1]

# 创建数据集，并将其划分为训练集和测试集
X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#################################################################################################

import math
import numpy as np

# 自定义朴素贝叶斯分类器类
# 自定义朴素贝叶斯分类器
class NaiveBayes:
    def __init__(self):
        self.model = None

    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'GaussianNB train done!'

    # 计算每个类别的概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    def predict(self, X_test):
        # 获取最大概率对应的类别标签
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))

# 初始化并训练自定义模型
model = NaiveBayes()
model.fit(X_train, y_train)

#################################################################################################

import matplotlib.pyplot as plt
import numpy as np

# 绘制决策边界
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    if isinstance(model, NaiveBayes):
        Z = np.array([model.predict(np.array([a, b])) for a, b in zip(xx.ravel(), yy.ravel())])
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('花萼长度')
    plt.ylabel('花萼宽度')
    plt.show()

# 绘制自定义模型的决策边界
plot_decision_boundary(model, X_train, y_train, "自定义朴素贝叶斯分类器的决策边界")

# 使用 sklearn 的 GaussianNB 进行对比
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
plot_decision_boundary(gnb, X_train, y_train, "sklearn GaussianNB 分类器的决策边界")