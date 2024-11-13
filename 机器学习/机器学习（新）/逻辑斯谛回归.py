import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])  # 只取前100个样本
    return data[:, :2], data[:, -1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class LogisticRegressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def data_matrix(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        data_mat = self.data_matrix(X)
        self.weights = np.zeros((data_mat.shape[1], 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
        print('LogisticRegression模型(学习率={}, 最大迭代次数={})'.format(self.learning_rate, self.max_iter))

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            predicted_label = 1 if result > 0 else 0
            if predicted_label == y:
                right += 1
        return right / len(X_test)

lr_clf = LogisticRegressionClassifier()
lr_clf.fit(X_train, y_train)

accuracy = lr_clf.score(X_test, y_test)
print(f'自定义模型精度: {accuracy:.2f}')

x_points = np.arange(4, 8)
y_ = -(lr_clf.weights[1] * x_points + lr_clf.weights[0]) / lr_clf.weights[2]
plt.plot(x_points, y_)

plt.scatter(X[:50, 0], X[:50, 1], label='0')
plt.scatter(X[50:, 0], X[50:, 1], label='1')
plt.legend()
plt.show()

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
sklearn_accuracy = clf.score(X_test, y_test)
print(f'Sklearn模型准确性: {sklearn_accuracy:.2f}')
print('系数:', clf.coef_, '截距:', clf.intercept_)

x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_[0]) / clf.coef_[0][1]  # 修改为 clf.intercept_[0]
plt.plot(x_points, y_)

plt.scatter(X[:50, 0], X[:50, 1], color='blue', label='0')
plt.scatter(X[50:, 0], X[50:, 1], color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
