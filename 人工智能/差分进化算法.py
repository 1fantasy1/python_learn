import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']

class Population:
    def __init__(self, NP=1000, size=10, xMin=-10, xMax=10, F=0.5, CR=0.8):
        self.NP = NP  # 种群规模
        self.size = size  # 个体的长度
        self.xMin = xMin  # 最小值
        self.xMax = xMax  # 最大值
        self.F = F  # 变异的控制参数
        self.CR = CR  # 杂交的控制参数

        self.X = [[0] * size for _ in range(NP)]  # 个体
        self.XMutation = [[0] * size for _ in range(NP)]
        self.XCrossOver = [[0] * size for _ in range(NP)]
        self.fitness_X = [0] * NP  # 适应值
        self.best_fitness = []  # 用来记录每一代的最优适应值

    # 获取种群个体
    def get_X(self):
        return self.X

    # 设置种群个体
    def set_X(self, XTemp):
        self.X = [row[:] for row in XTemp]

    # 获取适应值
    def get_fitness_X(self):
        return self.fitness_X

    # 设置适应值
    def set_fitness_X(self, fitness_X):
        self.fitness_X = fitness_X[:]

    # 获取变异后的种群
    def get_XMutation(self):
        return self.XMutation

    # 设置变异后的种群
    def set_XMutation(self, XMutationTemp):
        self.XMutation = [row[:] for row in XMutationTemp]

    # 获取交叉后的种群
    def get_XCrossOver(self):
        return self.XCrossOver

    # 设置交叉后的种群
    def set_XCrossOver(self, XCrossOverTemp):
        self.XCrossOver = [row[:] for row in XCrossOverTemp]

    # 适应值计算：个体的平方和
    def calculate_fitness(self, XTemp):
        return sum(x ** 2 for x in XTemp)

    # 初始化种群，计算适应值
    def initialize(self):
        XTemp = [[random.uniform(self.xMin, self.xMax) for _ in range(self.size)] for _ in range(self.NP)]
        FitnessTemp = [self.calculate_fitness(individual) for individual in XTemp]

        self.set_X(XTemp)
        self.set_fitness_X(FitnessTemp)

    # 变异操作
    def mutation(self):
        XTemp = self.get_X()
        XMutationTemp = [[0] * self.size for _ in range(self.NP)]

        for i in range(self.NP):
            r1, r2, r3 = random.sample([x for x in range(self.NP) if x != i], 3)
            for j in range(self.size):
                XMutationTemp[i][j] = XTemp[r1][j] + self.F * (XTemp[r2][j] - XTemp[r3][j])

        self.set_XMutation(XMutationTemp)

    # 交叉操作
    def crossover(self):
        XTemp = self.get_X()
        XMutationTemp = self.get_XMutation()
        XCrossOverTemp = [[0] * self.size for _ in range(self.NP)]

        for i in range(self.NP):
            for j in range(self.size):
                rTemp = random.random()
                if rTemp <= self.CR:
                    XCrossOverTemp[i][j] = XMutationTemp[i][j]
                else:
                    XCrossOverTemp[i][j] = XTemp[i][j]

        self.set_XCrossOver(XCrossOverTemp)

    # 选择操作：贪婪选择策略
    def selection(self):
        XTemp = self.get_X()
        XCrossOverTemp = self.get_XCrossOver()
        FitnessTemp = self.get_fitness_X()

        for i in range(self.NP):
            FitnessCrossOverTemp = self.calculate_fitness(XCrossOverTemp[i])
            if FitnessCrossOverTemp < FitnessTemp[i]:
                XTemp[i] = XCrossOverTemp[i]
                FitnessTemp[i] = FitnessCrossOverTemp

        self.set_X(XTemp)
        self.set_fitness_X(FitnessTemp)

    # 保存每一代的全局最优值
    def save_best(self):
        FitnessTemp = self.get_fitness_X()
        best_idx = min(range(self.NP), key=lambda i: FitnessTemp[i])
        self.best_fitness.append(FitnessTemp[best_idx])
        # print(FitnessTemp[best_idx])

    # 绘制最优适应值变化图
    def plot_best_fitness(self):
        plt.plot(range(len(self.best_fitness)), self.best_fitness, label='最佳适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.title('最佳适应度与代数的关系')
        plt.legend()
        plt.show()


class DETest:
    def __init__(self, max_cycle=1000):
        self.gen = 0  # 当前代数
        self.max_cycle = max_cycle  # 最大循环次数
        self.population = Population(NP=100, size=10, xMin=-10, xMax=10, F=0.5, CR=0.8)

    def run(self):
        self.population.initialize()  # 初始化种群
        while self.gen <= self.max_cycle:
            self.population.mutation()  # 变异操作
            self.population.crossover()  # 交叉操作
            self.population.selection()  # 选择操作
            self.gen += 1
            self.population.save_best()  # 输出每一代的最优适应值

        self.population.plot_best_fitness()  # 绘制最优适应值的变化曲线


if __name__ == "__main__":
    test = DETest(max_cycle=1000)  # 创建DETest对象，设置最大循环次数为1000
    test.run()  # 运行测试