import numpy as np
import random
# import pandas as pd

def load_data():
    inp = input()
    data = []
    while (inp):
        dx = [float(di) for di in inp.strip().split(',')]
        data.append(dx)
        inp = input()
    data = np.array(data)
    return data

# def load_data(filepath):
#     """
#     从文件中加载数据集。
#     参数：
#         filepath: 数据文件路径
#     返回值：
#         data: 数据集（DataFrame形式）
#     """
#     data = pd.read_csv(filepath)
#     return data

def data_split(X, y, test_size=0.4, random_state=5):
    n_samples = len(X)
    assert len(X) == len(y)

    indices = np.arange(n_samples)
    random.seed(random_state)

    train_indexs = list(set(random.sample(indices.tolist(), int(n_samples * (1 - test_size)))))
    test_indexs = [k for k in indices if k not in train_indexs]
    return X[train_indexs, :], X[test_indexs, :], y[train_indexs], y[test_indexs]


def fit(X, y):
    shape_x = X.shape
    dim = shape_x[1]
    w = np.zeros(shape=(dim, 1))
    b = 0
    done = False
    y = y.reshape(-1, 1)  # 转换y为列向量
    lr = 1  # 学习率（调整为1以匹配题目结果比例）
    max_iter = 1000  # 最大迭代次数
    for _ in range(max_iter):
        done = True  # 假设本次迭代没有误分类
        for i in range(shape_x[0]):
            if y[i] * (np.dot(X[i], w) + b) <= 0:  # 如果分类错误
                w += lr * y[i] * X[i].reshape(-1, 1)  # 更新权重
                b += lr * y[i][0]  # 更新偏置项
                done = False  # 存在误分类点，设置done为False
        if done:  # 如果没有误分类点，提前终止
            break
    return w, b


def predict(X, y, w, b):
    y_hat = np.sign(X.dot(w) + b)
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

# filepath = "E:/文档/文件测试/iris_dat.csv"
# data = load_data(filepath)
data = load_data()
X = data[:, :2]
y = data[:, -1]
# X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

X_train, X_test, y_train, y_test = data_split(X, y)
w, b = fit(X_train, y_train)
acc = predict(X_test, y_test, w, b)

print(w)
print(b)
print(acc)