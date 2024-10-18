import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()

# 将数据集转换为 Pandas DataFrame，并给每列命名
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 在 DataFrame 中新增一列 'label'，用于存放分类标签（目标值）
df['label'] = iris.target

# 重命名列名，使其更加简洁（去掉单位描述）
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# 绘制散点图，查看不同类别鸢尾花在萼片长度和萼片宽度上的分布
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')  # 类别 0（山鸢尾）
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')  # 类别 1（变色鸢尾）
plt.xlabel('sepal length')  # x轴标签
plt.ylabel('sepal width')   # y轴标签
plt.legend()  # 显示图例

# 将前 100 条数据（仅两类鸢尾花，线性可分）转换为 NumPy 数组，并提取特征和标签
data = np.array(df.iloc[:100, [0, 1, -1]])  # 取萼片长度、宽度及标签
X, y = data[:, :-1], data[:, -1]  # 特征 X（萼片长度、宽度），标签 y
y = np.array([1 if i == 1 else -1 for i in y])  # 将标签转换为 1 和 -1

# 定义感知器模型（Perceptron Model），用于二分类任务
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)  # 初始化权重向量
        self.b = 0  # 初始化偏置
        self.l_rate = 0.1  # 学习率

    # 定义符号函数，计算决策函数值
    def sign(self, x, w, b):
        return np.dot(x, w) + b

    # 使用随机梯度下降法（SGD）进行模型训练
    def fit(self, X_train, y_train):
        is_wrong = False  # 标志是否存在分类错误
        while not is_wrong:
            wrong_count = 0  # 记录分类错误次数
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                # 如果误分类，更新权重和偏置
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)  # 更新权重
                    self.b = self.b + self.l_rate * y  # 更新偏置
                    wrong_count += 1  # 记录一次错误分类
            if wrong_count == 0:  # 如果无误分类，结束训练
                is_wrong = True
        print('Training complete: Perceptron Model!')  # 输出训练完成信息

    # 评分方法，后续可以实现模型在测试集上的评分逻辑
    def score(self, X_test, y_test):
        pass

# 训练感知器模型
perceptron = Model()
perceptron.fit(X, y)

# 绘制决策边界（通过感知器模型的权重和偏置计算）
x_points = np.linspace(4, 7, 10)  # 在 x 轴上生成 10 个点
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]  # 计算相应的 y 值
plt.plot(x_points, y_)  # 绘制感知器决策边界

# 绘制原始数据的散点图
plt.plot(data[:50, 0], data[:50, 1], 'o', color='blue', label='0')  # 类别 0
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color='orange', label='1')  # 类别 1
plt.xlabel('sepal length')  # x轴标签
plt.ylabel('sepal width')   # y轴标签
plt.legend()  # 显示图例

# 使用 sklearn 的 Perceptron 类进行训练
from sklearn.linear_model import Perceptron
clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)  # 创建 Perceptron 模型，关闭拦截项
clf.fit(X, y)  # 训练模型
print(clf.coef_)  # 输出特征的权重
print(clf.intercept_)  # 输出截距（此处为 0，因为关闭了 intercept）

# 绘制 sklearn Perceptron 模型的决策边界
x_ponits = np.arange(4, 8)  # 在 x 轴上生成点
y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]  # 计算对应的 y 值
plt.plot(x_ponits, y_)  # 绘制决策边界

# 绘制原始数据的散点图
plt.plot(data[:50, 0], data[:50, 1], 'o', color='blue', label='0')  # 类别 0
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color='orange', label='1')  # 类别 1
plt.xlabel('sepal length')  # x轴标签
plt.ylabel('sepal width')   # y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形
