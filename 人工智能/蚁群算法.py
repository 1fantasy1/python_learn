import numpy as np


class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay=0.1, alpha=1, beta=2):
        """
        初始化蚁群算法
        distances: 距离矩阵
        n_ants: 蚂蚁数量
        n_iterations: 迭代次数
        decay: 信息素衰减率
        alpha: 信息素重要程度
        beta: 启发式因子重要程度
        """
        self.distances = distances
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        # 初始化信息素矩阵
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        self.best_path = None
        self.best_distance = float('inf')

    def run(self):
        for iteration in range(self.n_iterations):
            paths = self.construct_solutions()
            self.update_pheromone(paths)

            # 更新最优解
            for path in paths:
                distance = self.calculate_total_distance(path)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path

        return self.best_path, self.best_distance

    def construct_solutions(self):
        paths = []
        for ant in range(self.n_ants):
            path = self.construct_path()
            paths.append(path)
        return paths

    def construct_path(self):
        start_city = np.random.randint(self.n_cities)
        unvisited = list(range(self.n_cities))
        unvisited.remove(start_city)
        path = [start_city]

        while unvisited:
            current_city = path[-1]
            probabilities = self.calculate_probabilities(current_city, unvisited)
            next_city = np.random.choice(unvisited, p=probabilities)
            path.append(next_city)
            unvisited.remove(next_city)

        return path

    def calculate_probabilities(self, current_city, unvisited):
        pheromone = np.array([self.pheromone[current_city][j] for j in unvisited])
        distance = np.array([1.0 / self.distances[current_city][j] for j in unvisited])

        probabilities = (pheromone ** self.alpha) * (distance ** self.beta)
        probabilities = probabilities / probabilities.sum()

        return probabilities

    def update_pheromone(self, paths):
        # 信息素衰减
        self.pheromone = (1 - self.decay) * self.pheromone

        # 添加新的信息素
        for path in paths:
            distance = self.calculate_total_distance(path)
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1.0 / distance
            # 处理从最后一个城市到第一个城市的路径
            self.pheromone[path[-1]][path[0]] += 1.0 / distance

    def calculate_total_distance(self, path):
        total = 0
        for i in range(len(path) - 1):
            total += self.distances[path[i]][path[i + 1]]
        total += self.distances[path[-1]][path[0]]  # 返回起点
        return total

# 使用示例
if __name__ == "__main__":
    # 输入距离矩阵（4个城市）
    distances = np.array([
        [0, 2, 5, 7],
        [2, 0, 4, 3],
        [5, 4, 0, 6],
        [7, 3, 6, 0]
    ])

    # 创建ACO实例
    aco = AntColony(
        distances=distances,
        n_ants=5,
        n_iterations=100,
        decay=0.1,
        alpha=1,
        beta=2
    )

    # 运行算法
    best_path, best_distance = aco.run()

    print("最优路径:", best_path)
    print("最短距离:", best_distance)
