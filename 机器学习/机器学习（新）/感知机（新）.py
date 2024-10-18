# 加载鸢尾花数据集，并转换为DataFrame格式
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 加载数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target  # 添加label列作为目标值
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']  # 简化列名

# 提取前100个样本（仅两类鸢尾花），并将标签转换为-1和1
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]  # X是特征，y是标签
y = np.where(y == 1, 1, -1)  # 将标签从0,1转换为-1和1

#################################################################################################

# 自定义感知器模型类
class PerceptronModel:
    def __init__(self, learning_rate=0.1):
        self.w = np.ones(X.shape[1], dtype=np.float32)  # 初始化权重为1
        self.b = 0  # 初始化偏置为0
        self.l_rate = learning_rate  # 学习率

    # 感知器决策函数
    def sign(self, x):
        return np.dot(x, self.w) + self.b

    # 训练感知器模型
    def fit(self, X_train, y_train):
        while True:
            wrong_count = 0  # 记录误分类次数
            for i in range(len(X_train)):
                if y_train[i] * self.sign(X_train[i]) <= 0:
                    self.w += self.l_rate * y_train[i] * X_train[i]  # 更新权重
                    self.b += self.l_rate * y_train[i]  # 更新偏置
                    wrong_count += 1  # 记录误分类
            if wrong_count == 0:
                break
        print("训练完成：自定义感知器模型!")

#################################################################################################

from sklearn.linear_model import Perceptron
# 训练自定义感知器模型
custom_perceptron = PerceptronModel()
custom_perceptron.fit(X, y)  # 训练模型

# 使用sklearn的Perceptron类进行训练
sklearn_perceptron = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
sklearn_perceptron.fit(X, y)  # 训练sklearn感知器模型

#################################################################################################

import matplotlib.pyplot as plt

# 定义绘制决策边界的函数，针对不同模型做区分
def plot_decision_boundary(model, X, label, color, title=None, is_sklearn=False):
    x_points = np.linspace(4, 7, 10)
    if is_sklearn:
        y_boundary = -(model.coef_[0][0] * x_points + model.intercept_) / model.coef_[0][1]
    else:
        y_boundary = -(model.w[0] * x_points + model.b) / model.w[1]
    plt.plot(x_points, y_boundary, color=color, label=label)
    if title:
        plt.title(title)

# 绘制散点图
def plot_scatter(X, y, color0='blue', color1='orange'):
    plt.scatter(X[:50, 0], X[:50, 1], color=color0, label='0')
    plt.scatter(X[50:100, 0], X[50:100, 1], color=color1, label='1')
    plt.xlabel('花萼长度')
    plt.ylabel('萼片宽度')
    plt.legend()

# 绘制自定义感知器的决策边界
plt.figure(figsize=(10, 6))
plot_scatter(X, y)
plot_decision_boundary(custom_perceptron, X, '自定义感知器', 'red')

# 绘制sklearn感知器的决策边界
plot_decision_boundary(sklearn_perceptron, X, 'Sklearn感知器', 'green', title='决策边界对比', is_sklearn=True)

plt.legend()
plt.show()

