import random

# 遗传算法参数
POPULATION_SIZE = 100  # 种群大小
CHROMOSOME_LENGTH = 10  # 染色体长度
MAX_GENERATION = 100  # 最大迭代次数
MUTATION_RATE = 0.1  # 变异率
CROSSOVER_RATE = 0.8  # 交叉率


# 适应度函数（目标函数）
def fitness_function(x):
    """
    求解目标函数 f(x) = x^2
    """
    return x ** 2


# 解码函数
def decode_chromosome(chromosome):
    """
    将二进制染色体解码为实数
    """
    return int(''.join(map(str, chromosome)), 2) / (2 ** CHROMOSOME_LENGTH - 1) * 10


# 初始化种群
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]
        population.append(chromosome)
    return population


# 选择操作（轮盘赌选择）
def selection(population):
    """
    根据适应度进行选择
    """
    # 计算每个个体的适应度
    fitness_values = []
    for chromosome in population:
        x = decode_chromosome(chromosome)
        fitness_values.append(fitness_function(x))

    # 计算总适应度
    total_fitness = sum(fitness_values)

    # 计算选择概率
    selection_probs = [f / total_fitness for f in fitness_values]

    # 轮盘赌选择
    new_population = []
    for _ in range(POPULATION_SIZE):
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(selection_probs):
            cumulative_prob += prob
            if r <= cumulative_prob:
                new_population.append(population[i].copy())
                break

    return new_population


# 交叉操作
def crossover(population):
    """
    单点交叉
    """
    new_population = population.copy()

    for i in range(0, POPULATION_SIZE, 2):
        if random.random() < CROSSOVER_RATE:
            # 随机选择交叉点
            crossover_point = random.randint(1, CHROMOSOME_LENGTH - 1)

            # 交换染色体的部分
            new_population[i][:crossover_point], new_population[i + 1][:crossover_point] = \
                new_population[i + 1][:crossover_point], new_population[i][:crossover_point]

    return new_population


# 变异操作
def mutation(population):
    """
    基因变异
    """
    for chromosome in population:
        for j in range(CHROMOSOME_LENGTH):
            if random.random() < MUTATION_RATE:
                chromosome[j] = 1 - chromosome[j]

    return population


# 主遗传算法
def genetic_algorithm():
    # 初始化种群
    population = initialize_population()

    # 迭代进化
    for generation in range(MAX_GENERATION):
        # 选择
        population = selection(population)

        # 交叉
        population = crossover(population)

        # 变异
        population = mutation(population)

        # 找出最优个体
        best_chromosome = max(population, key=lambda x: fitness_function(decode_chromosome(x)))
        best_fitness = fitness_function(decode_chromosome(best_chromosome))

        print(f"第{generation + 1}代: 最佳适应度={best_fitness}")

    # 返回最优解
    best_chromosome = max(population, key=lambda x: fitness_function(decode_chromosome(x)))
    best_x = decode_chromosome(best_chromosome)

    return best_x, fitness_function(best_x)


# 运行遗传算法
if __name__ == "__main__":
    best_solution, best_fitness = genetic_algorithm()
    print(f"\n最优解: x = {best_solution}")
    print(f"最佳适应度: f(x) = {best_fitness}")
