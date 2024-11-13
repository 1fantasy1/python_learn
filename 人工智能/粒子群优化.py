import numpy as np


# 定义目标函数
def objective_function(x):
    return x ** 2


# PSO 算法参数
num_particles = 30  # 粒子数目
dim = 1  # 搜索空间维度，这里是1维的
max_iter = 100  # 最大迭代次数
w = 0.5  # 惯性权重
c1 = 1.5  # 个人经验权重
c2 = 1.5  # 群体经验权重
lower_bound = -10  # 搜索空间下界
upper_bound = 10  # 搜索空间上界

# 初始化粒子位置和速度
positions = np.random.uniform(lower_bound, upper_bound, (num_particles, dim))  # 粒子的位置
velocities = np.random.uniform(-1, 1, (num_particles, dim))  # 粒子的速度
personal_best_positions = np.copy(positions)  # 每个粒子的历史最优位置
personal_best_scores = np.array([objective_function(p) for p in positions])  # 每个粒子的历史最优适应度

# 全局最优解初始化
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# PSO 算法主循环
for iteration in range(max_iter):
    # 更新速度和位置
    r1 = np.random.rand(num_particles, dim)
    r2 = np.random.rand(num_particles, dim)

    velocities = (w * velocities +
                  c1 * r1 * (personal_best_positions - positions) +
                  c2 * r2 * (global_best_position - positions))

    positions = positions + velocities  # 更新粒子的位置

    # 处理边界问题，确保粒子不会超出搜索空间范围
    positions = np.clip(positions, lower_bound, upper_bound)

    # 计算当前粒子的适应度
    scores = np.array([objective_function(p) for p in positions])

    # 更新每个粒子的历史最优解
    for i in range(num_particles):
        if scores[i] < personal_best_scores[i]:
            personal_best_scores[i] = scores[i]
            personal_best_positions[i] = positions[i]

    # 更新全局最优解
    min_score_index = np.argmin(personal_best_scores)
    if personal_best_scores[min_score_index] < global_best_score:
        global_best_score = personal_best_scores[min_score_index]
        global_best_position = personal_best_positions[min_score_index]

    # 输出当前迭代的信息
    print(f"第 {iteration + 1} 轮迭代, 当前最优适应度: {global_best_score}")

# 最终输出全局最优解
print("\n全局最优位置:", global_best_position)
print("全局最优适应度:", global_best_score)