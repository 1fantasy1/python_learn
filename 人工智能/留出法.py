import random
import numpy as np


def train_test_split(X, test_size=0.2, random_state=5):
    random.seed(random_state)
    n_samples = len(X)  # 数据集总样本数
    n_test = int(n_samples * test_size)  # 计算测试集样本数
    indices = list(range(n_samples))  # 生成数据索引
    random.shuffle(indices)  # 随机打乱索引

    # 分割训练集和测试集索引
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # 根据索引划分训练集和测试集
    train_X = X[train_indices]
    test_X = X[test_indices]

    return train_X, test_X


test_size = 0.2
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train_X, test_X = train_test_split(X, test_size=test_size)
print(train_X, test_X)

print("debug_begin");
print(len(test_X) == int(len(X) * test_size))
print("debug_end");