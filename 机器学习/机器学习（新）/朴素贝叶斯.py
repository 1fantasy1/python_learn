# 导入必要的库
import math  # 数学计算模块
import numpy as np  # 科学计算库，提供数组和矩阵操作
import pandas as pd  # 数据分析和处理库
import matplotlib.pyplot as plt  # 可视化绘图库
from sklearn.naive_bayes import GaussianNB  # sklearn 的高斯朴素贝叶斯分类器
from sklearn.model_selection import train_test_split  # 数据集划分工具
from matplotlib import rcParams  # Matplotlib 的运行时配置参数

# 设置 Matplotlib 的字体参数，防止中文显示乱码
rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 定义创建数据集的函数
def create_data():
    # 从 sklearn.datasets 导入鸢尾花数据集
    from sklearn.datasets import load_iris
    iris = load_iris()  # 加载鸢尾花数据集
    # 将数据转换为 DataFrame 格式，便于数据处理
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target  # 添加标签列
    # 重命名列名，方便后续处理
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # 选择前 150 行数据，并转换为 NumPy 数组
    data = np.array(df.iloc[:150, :])
    # 返回特征和标签，这里只选择前两列特征（花萼长度和宽度），以便于后续的可视化
    return data[:, :2], data[:, -1]

# 获取数据集
X, y = create_data()
# 将数据集划分为训练集和测试集，测试集占 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 自定义朴素贝叶斯分类器
class NaiveBayes:
    def __init__(self):
        self.model = None  # 初始化模型参数

    @staticmethod
    def mean(X):
        # 计算给定数据集的均值
        return sum(X) / float(len(X))

    def stdev(self, X):
        # 计算给定数据集的标准差
        avg = self.mean(X)
        # 计算每个数据点与均值的差的平方和，然后取平方根
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    def gaussian_probability(self, x, mean, stdev):
        # 计算高斯概率密度函数值
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        # 返回概率密度值
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self, train_data):
        # 对训练数据的每个特征计算均值和标准差
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        # 返回每个特征的均值和标准差
        return summaries

    def fit(self, X, y):
        # 训练模型，计算各类别下每个特征的均值和标准差
        labels = list(set(y))  # 获取所有类别的标签
        data = {label: [] for label in labels}  # 初始化一个字典，存储每个类别的数据
        for f, label in zip(X, y):
            data[label].append(f)  # 将对应类别的数据添加到字典中
        # 对每个类别的数据进行统计，计算均值和标准差
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'GaussianNB train done!'  # 返回训练完成的信息

    def calculate_probabilities(self, input_data):
        # 计算输入数据属于各个类别的概率
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1  # 初始化概率为 1
            for i in range(len(value)):
                mean, stdev = value[i]  # 获取第 i 个特征的均值和标准差
                x = input_data[i]  # 获取输入数据的第 i 个特征值
                # 使用高斯概率密度函数计算概率，并累乘（朴素贝叶斯假设特征之间相互独立）
                probabilities[label] *= self.gaussian_probability(x, mean, stdev)
        return probabilities  # 返回所有类别的概率

    def predict(self, X_test):
        # 对单个样本进行预测
        probabilities = self.calculate_probabilities(X_test)
        # 根据概率大小进行排序，取概率最大的类别作为预测结果
        label = sorted(probabilities.items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        # 计算模型在测试集上的准确率
        right = 0  # 记录预测正确的样本数量
        for X, y in zip(X_test, y_test):
            label = self.predict(X)  # 对样本进行预测
            if label == y:
                right += 1  # 如果预测正确，计数加一
        # 返回准确率
        return right / float(len(X_test))

# 初始化自定义的朴素贝叶斯分类器
model = NaiveBayes()

# 训练自定义模型
model.fit(X_train, y_train)

# 使用 sklearn 的 GaussianNB 进行对比
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 定义绘制决策边界的函数
def plot_decision_boundary(model, X, y, title):
    # 设置网格的范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 第一个特征的范围
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 第二个特征的范围
    # 生成网格点坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    # 对网格中的每个点进行分类预测
    if isinstance(model, NaiveBayes):
        # 如果是自定义的模型，需要手动遍历预测
        Z = np.array([model.predict(np.array([a, b])) for a, b in zip(xx.ravel(), yy.ravel())])
    else:
        # 如果是 sklearn 的模型，可以直接使用 predict 方法
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # 将结果 reshaped 成与网格相同的形状

    # 绘制决策边界（等高线图）
    plt.contourf(xx, yy, Z, alpha=0.8)
    # 绘制训练数据的散点图
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    # 设置标题和坐标轴标签
    plt.title(title)
    plt.xlabel('花萼长度')
    plt.ylabel('花萼宽度')
    plt.show()  # 显示图形

# 绘制自定义模型的决策边界
plot_decision_boundary(model, X_train, y_train, "自定义朴素贝叶斯分类器的决策边界")

# 绘制 sklearn 模型的决策边界
plot_decision_boundary(gnb, X_train, y_train, "sklearn 朴素贝叶斯分类器的决策边界")

# 比较分类效果
sklearn_accuracy = gnb.score(X_test, y_test)  # 计算 sklearn 模型的准确率
custom_accuracy = model.score(X_test, y_test)  # 计算自定义模型的准确率
# 打印准确率
print(f"sklearn 朴素贝叶斯分类器的准确率：{sklearn_accuracy:.2f}")
print(f"自定义朴素贝叶斯分类器的准确率：{custom_accuracy:.2f}")
