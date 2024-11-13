import numpy as np
import random
import math


# 计算两点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 计算每个粒子路径的总距离（适应度函数）
def calculate_fitness(route, distance_matrix, num_vehicles, capacity, demands):
    total_distance = 0
    vehicle_capacity = [0] * num_vehicles  # 每辆车的当前负载

    # 车辆路径
    current_vehicle = 0
    current_load = 0
    previous_customer = 0  # 配送中心（假设是第0号客户）

    for customer in route:
        customer = int(np.floor(customer))  # 确保customer是整数索引
        if current_load + demands[customer] > capacity:
            # 如果超出容量限制，则分配到新的车辆
            current_vehicle += 1
            current_load = 0
            if current_vehicle >= num_vehicles:
                return float('inf')  # 如果车辆不够，则返回一个较差的适应度
        current_load += demands[customer]
        total_distance += distance_matrix[previous_customer][customer]
        previous_customer = customer

    # 车辆返回配送中心
    total_distance += distance_matrix[previous_customer][0]
    return total_distance


# 粒子群算法
def pso_vrp(num_customers, num_vehicles, capacity, distance_matrix, demands, num_particles=30, max_iterations=100,
            w=0.5, c1=1.5, c2=1.5):
    # 初始化粒子位置（每个粒子是一个路线的排列）
    particles = [np.random.permutation(num_customers) for _ in range(num_particles)]
    velocities = [np.zeros(num_customers) for _ in range(num_particles)]

    # 个体最优位置和全局最优位置
    pbest = particles[:]
    gbest = particles[
        np.argmin([calculate_fitness(p, distance_matrix, num_vehicles, capacity, demands) for p in particles])]

    # 适应度（总行驶距离）
    pbest_fitness = [calculate_fitness(p, distance_matrix, num_vehicles, capacity, demands) for p in particles]
    gbest_fitness = min(pbest_fitness)

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # 更新粒子速度
            velocities[i] = (w * velocities[i] + c1 * np.random.random() * (
                        pbest[i] - particles[i]) + c2 * np.random.random() * (gbest - particles[i]))

            # 更新粒子位置
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, num_customers - 1)  # 保证位置在合理范围内

            # 计算新位置的适应度
            fitness = calculate_fitness(particles[i], distance_matrix, num_vehicles, capacity, demands)

            if fitness < pbest_fitness[i]:
                # 更新个体最优
                pbest[i] = particles[i]
                pbest_fitness[i] = fitness

                if fitness < gbest_fitness:
                    # 更新全局最优
                    gbest = particles[i]
                    gbest_fitness = fitness

        print(f"第 {iteration + 1} 迭代, 最佳适应度:{gbest_fitness}")

    return gbest, gbest_fitness


# 测试数据
num_customers = 10  # 客户数量
num_vehicles = 3  # 车辆数量
capacity = 15  # 每辆车的最大容量
customers = [(random.uniform(0, 10), random.uniform(0, 10), random.randint(1, 5)) for _ in
             range(num_customers)]  # 客户位置和需求
distance_matrix = np.zeros((num_customers + 1, num_customers + 1))  # 距离矩阵，包括配送中心
demands = [customer[2] for customer in customers]  # 客户的需求量

# 计算距离矩阵
for i in range(num_customers + 1):
    for j in range(i + 1, num_customers + 1):
        if i == 0:  # 配送中心到客户
            distance_matrix[i][j] = distance_matrix[j][i] = euclidean_distance((0, 0),
                                                                               customers[j - 1][:2] if j != 0 else (
                                                                               0, 0))
        else:  # 客户到客户
            distance_matrix[i][j] = distance_matrix[j][i] = euclidean_distance(customers[i - 1][:2],
                                                                               customers[j - 1][:2])

# 使用粒子群算法解决VRP
best_route, best_fitness = pso_vrp(num_customers, num_vehicles, capacity, distance_matrix, demands)

print("最佳路径:", best_route)
print("最短总行驶距离:", best_fitness)
