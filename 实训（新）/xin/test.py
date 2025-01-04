# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import fetch_california_housing
# from tqdm import tqdm
#
# # 定义适应度函数
# def fitness_function(params, X_train, y_train, X_test, y_test):
#     n_estimators, min_samples_leaf = int(params[0]), int(params[1])
#     if n_estimators <= 0 or min_samples_leaf <= 0:
#         return float('inf')
#     rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=42)
#     rf.fit(X_train, y_train)
#     return mean_squared_error(y_test, rf.predict(X_test))  # 只考虑测试集的MSE
#
# # 简化版未来搜索算法 (FSA)
# def future_search_algorithm(X_train, y_train, X_test, y_test, num_countries=10, max_iterations=50):
#     dim = 2  # 参数维度
#     lb, ub = np.array([10, 1]), np.array([200, 10])  # 参数上下界
#     S = lb + (ub - lb) * np.random.rand(num_countries, dim)  # 初始化国家/地区
#     LS = S.copy()  # 局部最优解初始化为初始状态
#     GS = S[np.argmin([fitness_function(s, X_train, y_train, X_test, y_test) for s in S])].copy()  # 全局最优解
#
#     for iteration in tqdm(range(max_iterations), desc="FSA Iterations"):
#         for i in range(num_countries):
#             # 更新每个国家的位置
#             S[i] += (LS[i] - S[i]) * np.random.rand(dim) + (GS - S[i]) * np.random.rand(dim)
#             S[i] = np.clip(S[i], lb, ub)
#
#             # 计算新位置的适应度并更新局部最优解
#             new_fitness = fitness_function(S[i], X_train, y_train, X_test, y_test)
#             if new_fitness < fitness_function(LS[i], X_train, y_train, X_test, y_test):
#                 LS[i] = S[i].copy()
#
#         # 更新全局最优解
#         best_local = LS[np.argmin([fitness_function(ls, X_train, y_train, X_test, y_test) for ls in LS])]
#         if fitness_function(best_local, X_train, y_train, X_test, y_test) < fitness_function(GS, X_train, y_train, X_test, y_test):
#             GS = best_local.copy()
#
#         print(f"Iteration {iteration + 1}: Best MSE = {fitness_function(GS, X_train, y_train, X_test, y_test)}, Best Params = {GS}")
#
#     return GS
#
# # 使用加州房价数据集
# housing = fetch_california_housing()
# X, y = housing.data, housing.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# best_params = future_search_algorithm(X_train, y_train, X_test, y_test)
# print("Best parameters found:", best_params)
#
# # 使用最佳参数训练最终模型
# final_rf = RandomForestRegressor(n_estimators=int(best_params[0]), min_samples_leaf=int(best_params[1]), random_state=42)
# final_rf.fit(X_train, y_train)
# final_mse = mean_squared_error(y_test, final_rf.predict(X_test))
# print(f"Final MSE on test set: {final_mse}")

# import numpy as np
#
#
# class FSA:
#     def __init__(self, func, dim, pop_size, lb, ub, max_iter):
#         """
#         初始化FSA算法
#         func: 目标函数
#         dim: 问题维度
#         pop_size: 种群大小（国家/地区数量）
#         lb: 下界
#         ub: 上界
#         max_iter: 最大迭代次数
#         """
#         self.func = func
#         self.dim = dim
#         self.pop_size = pop_size
#         self.lb = lb if isinstance(lb, np.ndarray) else np.array([lb] * dim)
#         self.ub = ub if isinstance(ub, np.ndarray) else np.array([ub] * dim)
#         self.max_iter = max_iter
#
#         # 初始化最优解
#         self.best_solution = None
#         self.best_fitness = float('inf')
#
#     def initialize(self):
#         """
#         step1: 初始化国家/地区
#         根据式(1): S(i,:) = Lb + (Ub-Lb)*rand(1,d)
#         """
#         return self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
#
#     def evaluate(self, positions):
#         """
#         计算适应度值
#         """
#         return np.array([self.func(pos) for pos in positions])
#
#     def update_position(self, current_pos, local_best, global_best):
#         """
#         step3: 更新位置
#         根据式(3),(4),(5)更新位置
#         """
#         # 式(3): 局部解更新
#         S_L = (local_best - current_pos) * np.random.rand(self.dim)
#
#         # 式(4): 全局解更新
#         S_G = (global_best - current_pos) * np.random.rand(self.dim)
#
#         # 式(5): 更新当前解
#         new_pos = current_pos + S_L + S_G
#
#         # 确保在搜索范围内
#         new_pos = np.clip(new_pos, self.lb, self.ub)
#         return new_pos
#
#     def update_random_init(self, current_pos, local_best, global_best):
#         """
#         step5: 更新随机初始值
#         根据式(6): S(i,:) = GS + [GS-LS(i,:)]*rand
#         """
#         return global_best + (global_best - local_best) * np.random.rand(self.dim)
#
#     def optimize(self):
#         """
#         主优化循环
#         """
#         # 初始化种群
#         positions = self.initialize()
#         local_best_positions = positions.copy()
#         local_best_fitness = self.evaluate(positions)
#
#         # 初始化全局最优
#         global_best_idx = np.argmin(local_best_fitness)
#         global_best_position = positions[global_best_idx].copy()
#         global_best_fitness = local_best_fitness[global_best_idx]
#
#         # 迭代优化
#         for iter in range(self.max_iter):
#             # 更新每个个体的位置
#             for i in range(self.pop_size):
#                 # 更新位置
#                 new_position = self.update_position(
#                     positions[i],
#                     local_best_positions[i],
#                     global_best_position
#                 )
#
#                 # 评估新位置
#                 new_fitness = self.func(new_position)
#
#                 # 更新局部最优
#                 if new_fitness < local_best_fitness[i]:
#                     local_best_fitness[i] = new_fitness
#                     local_best_positions[i] = new_position.copy()
#
#                     # 更新全局最优
#                     if new_fitness < global_best_fitness:
#                         global_best_fitness = new_fitness
#                         global_best_position = new_position.copy()
#
#                 # 更新随机初始值
#                 positions[i] = self.update_random_init(
#                     positions[i],
#                     local_best_positions[i],
#                     global_best_position
#                 )
#
#             # 保存当前最优解
#             if global_best_fitness < self.best_fitness:
#                 self.best_fitness = global_best_fitness
#                 self.best_solution = global_best_position.copy()
#
#         return self.best_solution, self.best_fitness
#
#
# # 使用示例
# def test_function(x):
#     """
#     测试函数: 以简单的球函数为例
#     """
#     return np.sum(x ** 2)
#
#
# # 设置参数
# dim = 30  # 维度
# pop_size = 50  # 种群大小
# lb = -100  # 下界
# ub = 100  # 上界
# max_iter = 1000  # 最大迭代次数
#
# # 创建并运行FSA算法
# fsa = FSA(test_function, dim, pop_size, lb, ub, max_iter)
# best_solution, best_fitness = fsa.optimize()
#
# print("最优解:", best_solution)
# print("最优适应度值:", best_fitness)


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class FSA_RF_Optimizer:
    def __init__(self, X, y, pop_size=30, max_iter=100):
        """
        初始化FSA优化器
        X: 特征数据
        y: 目标变量
        pop_size: 种群大小
        max_iter: 最大迭代次数
        """
        # 数据集分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = 2  # 优化两个参数：n_estimators 和 min_samples_leaf

        # 参数范围设置
        self.lb = np.array([10, 1])  # n_estimators最小10，min_samples_leaf最小1
        self.ub = np.array([200, 20])  # n_estimators最大200，min_samples_leaf最大20

        # 初始化最优解
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize(self):
        """初始化种群"""
        # 确保生成的参数为整数
        positions = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        return np.round(positions)

    def evaluate(self, position):
        """
        评估函数：计算RF在训练集和测试集上的MSE之和
        """
        # 确保参数为整数
        n_estimators = int(position[0])
        min_samples_leaf = int(position[1])

        # 创建RF模型
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # 训练模型
        rf.fit(self.X_train, self.y_train)

        # 预测
        train_pred = rf.predict(self.X_train)
        test_pred = rf.predict(self.X_test)

        # 计算MSE
        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)

        return train_mse + test_mse

    def update_position(self, current_pos, local_best, global_best):
        """更新位置"""
        S_L = (local_best - current_pos) * np.random.rand(self.dim)
        S_G = (global_best - current_pos) * np.random.rand(self.dim)
        new_pos = current_pos + S_L + S_G

        # 确保在搜索范围内且为整数
        new_pos = np.clip(new_pos, self.lb, self.ub)
        return np.round(new_pos)

    def update_random_init(self, current_pos, local_best, global_best):
        """更新随机初始值"""
        new_pos = global_best + (global_best - local_best) * np.random.rand(self.dim)
        new_pos = np.clip(new_pos, self.lb, self.ub)
        return np.round(new_pos)

    def optimize(self):
        """主优化循环"""
        # 初始化种群
        positions = self.initialize()
        local_best_positions = positions.copy()
        local_best_fitness = np.array([self.evaluate(pos) for pos in positions])

        # 初始化全局最优
        global_best_idx = np.argmin(local_best_fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = local_best_fitness[global_best_idx]

        # 迭代优化
        for iter in range(self.max_iter):
            print(f"迭代 {iter + 1}/{self.max_iter}, 当前最优适应度: {global_best_fitness:.6f}")

            for i in range(self.pop_size):
                # 更新位置
                new_position = self.update_position(
                    positions[i],
                    local_best_positions[i],
                    global_best_position
                )

                # 评估新位置
                new_fitness = self.evaluate(new_position)

                # 更新局部最优
                if new_fitness < local_best_fitness[i]:
                    local_best_fitness[i] = new_fitness
                    local_best_positions[i] = new_position.copy()

                    # 更新全局最优
                    if new_fitness < global_best_fitness:
                        global_best_fitness = new_fitness
                        global_best_position = new_position.copy()

                # 更新随机初始值
                positions[i] = self.update_random_init(
                    positions[i],
                    local_best_positions[i],
                    global_best_position
                )

            # 保存当前最优解
            if global_best_fitness < self.best_fitness:
                self.best_fitness = global_best_fitness
                self.best_solution = global_best_position.copy()

        return {
            'n_estimators': int(self.best_solution[0]),
            'min_samples_leaf': int(self.best_solution[1]),
            'best_fitness': self.best_fitness
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    from sklearn.datasets import make_regression

    # 生成示例回归数据
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        noise=0.1,
        random_state=42
    )

    # 创建优化器
    optimizer = FSA_RF_Optimizer(X, y, pop_size=30, max_iter=50)

    # 运行优化
    best_params = optimizer.optimize()

    print("\n最优参数:")
    print(f"n_estimators: {best_params['n_estimators']}")
    print(f"min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"最优适应度(MSE): {best_params['best_fitness']:.6f}")

    # 使用最优参数训练最终模型
    final_rf = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )

    final_rf.fit(optimizer.X_train, optimizer.y_train)

    # 评估最终模型
    train_pred = final_rf.predict(optimizer.X_train)
    test_pred = final_rf.predict(optimizer.X_test)

    train_mse = mean_squared_error(optimizer.y_train, train_pred)
    test_mse = mean_squared_error(optimizer.y_test, test_pred)

    print("\n最终模型性能:")
    print(f"训练集MSE: {train_mse:.6f}")
    print(f"测试集MSE: {test_mse:.6f}")