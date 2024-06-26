import pandas as pd
import numpy as np
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
rcParams['font.sans-serif'] = ['Microsoft YaHei']

def main():
    # 加载数据
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

    # 分割数据，训练集:测试集 = 7:3
    x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

    # 标准化特征
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)

    # 计算协方差矩阵及其特征值和特征向量
    cov_matrix = np.cov(x_train_std.T)
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix)

    # 计算解释方差比率
    tot = sum(eigen_val)
    var_exp = [(i / tot) for i in sorted(eigen_val, reverse=True)]

    # 特征变换
    eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:, i]) for i in range(len(eigen_val))]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    x_train_pca = x_train_std.dot(w)

    # 绘制结果
    color = ['r', 'g', 'b']
    marker = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), color, marker):
        plt.scatter(x_train_pca[y_train == l, 0],
                    x_train_pca[y_train == l, 1],
                    c=c, label=l, marker=m)
    plt.title('葡萄酒数据集的PCA')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    main()