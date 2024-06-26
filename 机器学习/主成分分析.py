import numpy as np
from sklearn.decomposition import PCA
# 假设X是原始数据矩阵，每一行表示一个样本，每一列表示一个特征
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 创建PCA对象，n_components指定降维后的维度数
pca = PCA(n_components=2)
# 对数据进行PCA降维
X_pca = pca.fit_transform(X)
# 输出降维后的数据
print(X_pca)