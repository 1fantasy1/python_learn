import numpy as np
import random


class AdaptiveGeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, max_generations):
        """
        初始化遗传算法参数

        :param population_size: 种群大小
        :param chromosome_length: 染色体长度
        :param mutation_rate: 初始变异率
        :param crossover_rate: 交叉率
        :param max_generations: 最大迭代次数
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations

        # 自适应参数
        self.adaptive_mutation_rate = mutation_rate
        self.adaptive_crossover_rate = crossover_rate

    def initialize_population(self):
        """
        随机初始化种群

        :return: 初始种群
        """
        population = np.random.randint(2, size=(self.population_size, self.chromosome_length))
        return population

    def fitness_function(self, chromosome):
        """
        适应度函数（需要根据具体问题自定义）

        :param chromosome: 染色体
        :return: 适应度值
        """
        # 示例：求解一个简单的优化问题
        # 例如：寻找使目标函数最大的二进制染色体
        x = int(''.join(map(str, chromosome)), 2)
        return x * np.sin(x)

    def selection(self, population):
        """
        选择操作（轮盘赌选择）

        :param population: 当前种群
        :return: 选择后的种群
        """
        # 计算适应度
        fitness_values = np.array([self.fitness_function(chrom) for chrom in population])

        # 归一化适应度
        total_fitness = np.sum(fitness_values)
        selection_probs = fitness_values / total_fitness

        # 轮盘赌选择
        new_population = []
        for _ in range(self.population_size):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    new_population.append(population[i])
                    break

        return np.array(new_population)

    def crossover(self, population):
        """
        交叉操作（自适应交叉率）

        :param population: 当前种群
        :return: 交叉后的种群
        """
        new_population = population.copy()

        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                # 自适应交叉率
                if random.random() < self.adaptive_crossover_rate:
                    # 随机选择交叉点
                    crossover_point = random.randint(1, self.chromosome_length - 1)

                    # 进行交叉
                    new_population[i][:crossover_point], new_population[i + 1][:crossover_point] = \
                        new_population[i + 1][:crossover_point], new_population[i][:crossover_point]

        return new_population

    def mutation(self, population):
        """
        变异操作（自适应变异率）

        :param population: 当前种群
        :return: 变异后的种群
        """
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                # 自适应变异率
                if random.random() < self.adaptive_mutation_rate:
                    population[i][j] = 1 - population[i][j]

        return population

    def adaptive_parameter_adjustment(self, generation, best_fitness, mean_fitness):
        """
        自适应参数调整

        :param generation: 当前代数
        :param best_fitness: 最佳适应度
        :param mean_fitness: 平均适应度
        """
        # 根据最佳适应度和平均适应度动态调整变异率和交叉率
        progress_ratio = best_fitness / mean_fitness

        # 自适应变异率
        if progress_ratio < 1.0:
            self.adaptive_mutation_rate = min(self.mutation_rate * 1.5, 0.5)
        else:
            self.adaptive_mutation_rate = max(self.mutation_rate * 0.5, 0.01)

        # 自适应交叉率
        if progress_ratio < 1.0:
            self.adaptive_crossover_rate = max(self.crossover_rate * 0.5, 0.1)
        else:
            self.adaptive_crossover_rate = min(self.crossover_rate * 1.5, 0.9)

    def run(self):
        """
        运行遗传算法

        :return: 最优解
        """
        # 初始化种群
        population = self.initialize_population()

        best_solution = None
        best_fitness = float('-inf')

        # 迭代
        for generation in range(self.max_generations):
            # 评估种群
            fitness_values = [self.fitness_function(chrom) for chrom in population]
            current_best_fitness = max(fitness_values)
            mean_fitness = np.mean(fitness_values)

            # 更新最优解
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[fitness_values.index(current_best_fitness)]

            # 自适应参数调整
            self.adaptive_parameter_adjustment(generation, best_fitness, mean_fitness)

            # 选择
            population = self.selection(population)

            # 交叉
            population = self.crossover(population)

            # 变异
            population = self.mutation(population)

            # 打印当前代的最优解
            print(f"第{generation}代: 最佳适应度={best_fitness}")

        return best_solution, best_fitness


# 使用示例
def main():
    # 参数设置
    population_size = 100
    chromosome_length = 10
    mutation_rate = 0.01
    crossover_rate = 0.8
    max_generations = 50

    # 创建并运行遗传算法
    ga = AdaptiveGeneticAlgorithm(population_size, chromosome_length,
                                  mutation_rate, crossover_rate, max_generations)

    best_solution, best_fitness = ga.run()

    print("\n最优解:", best_solution)
    print("最佳适应度:", best_fitness)


if __name__ == "__main__":
    main()
