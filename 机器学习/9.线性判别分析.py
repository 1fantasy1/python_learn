# 线性判别分析
import numpy as np
from matplotlib import rcParams
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 线性判别分析降维，保留两个线性判别分量
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# 在降维后的数据上训练LDA分类器
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train_lda, y_train)

# 进行预测
y_pred = lda_classifier.predict(X_test_lda)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"分类准确率: {accuracy * 100:.2f}%")

# 可视化降维后的训练数据
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
markers = ['o', 's', 'd']
for i, color, marker in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == i, 0], X_train_lda[y_train == i, 1], color=color, label=iris.target_names[i], marker=marker)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='best')
plt.title('LDA: 鸢尾花训练集')
plt.show()