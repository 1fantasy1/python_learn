import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# 读取数据
data = pd.read_csv('C:\\audit_risk.csv')
X = data.iloc[:, :-1]  # 取所有行，除了最后一列
y = data.iloc[:, -1]   # 取所有行，最后一列作为标签

# 将非数值特征转换为数值特征
X = pd.get_dummies(X)

# 处理缺失值
imputer = SimpleImputer(strategy='mean')  # 这里使用均值插补，也可以选择其他策略如 'median', 'most_frequent', 'constant'
X = imputer.fit_transform(X)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 模型训练与预测
estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)

# 计算准确率
print(np.mean(y_test == y_predicted))