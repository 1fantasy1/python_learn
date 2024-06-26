import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    X: 数据点的集合，形状为 (n_samples, n_features)
    k: 聚类数量
    max_iters: 最大迭代次数
    tol: 终止条件阈值
    """
    n_samples, n_features = X.shape

    # 随机初始化聚类中心
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for i in range(max_iters):
        # 计算每个点到所有聚类中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # 分配每个点到最近的聚类中心
        labels = np.argmin(distances, axis=1)

        # 计算新的聚类中心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # 检查是否收敛
        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return centroids, labels

# 示例使用
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # 生成示例数据
    X, _ = make_blobs(n_samples=2000, centers=9, random_state=40)

    # 执行 k-means 算法
    centroids, labels = kmeans(X, k=9)

    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=30, c='red')
    plt.show()