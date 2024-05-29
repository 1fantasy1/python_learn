import numpy as np

class DualSimplex(object):
    def __init__(self, objective_coefficients, constraint_coefficients):
        self.num_variables = len(objective_coefficients)  # 变量个数
        self.num_constraints = len(constraint_coefficients)  # 约束条件个数
        self.objective_coefficients = objective_coefficients  # 目标函数系数
        self.C = []  # 检验数
        self.constraint_coefficients = constraint_coefficients  # 约束条件系数矩阵
        self.basic_variables = list(range(self.num_constraints))  # 初始化基变量
        self.solution_found = False

    def check_optimality(self):
        self.solution_found = all(self.constraint_coefficients[i][-1] >= 0 for i in range(self.num_constraints))

    def solve(self):
        max_iterations = 100  # 防止无限迭代
        while max_iterations > 0:
            self.C = []  # 清空检验数
            for j in range(self.num_variables):
                zj = sum(self.constraint_coefficients[i][j] * self.objective_coefficients[self.basic_variables[i]]
                         for i in range(self.num_constraints))  # 计算检验数
                self.C.append(self.objective_coefficients[j] - zj)

            self.check_optimality()
            if self.solution_found:
                break
            else:
                self.pivot()  # 进行基变换
                max_iterations -= 1

        self.print_solution()

    def pivot(self):
        if np.min(self.constraint_coefficients[:, :-1]) >= 0:
            self.special = True
        bi = [self.constraint_coefficients[i][-1] for i in range(self.num_constraints)]
        leaving_variable_index = bi.index(min(bi))  # 确定出基变量
        ratios = []
        for j in range(self.num_variables):
            if self.constraint_coefficients[leaving_variable_index][j] <= 0 or self.C[j] == 0:
                ratio = float('inf')
            else:
                ratio = self.C[j] / self.constraint_coefficients[leaving_variable_index][j]
            ratios.append(ratio)
        entering_variable_index = ratios.index(min(ratios))  # 确定进基变量
        pivot_element = self.constraint_coefficients[leaving_variable_index][entering_variable_index]

        self.basic_variables[leaving_variable_index] = entering_variable_index  # 更新基变量
        for x in range(self.num_variables + 1):
            self.constraint_coefficients[leaving_variable_index][x] /= pivot_element
        for k in range(self.num_constraints):
            if k != leaving_variable_index:
                multiplier = self.constraint_coefficients[k][entering_variable_index]  # 倍数
                for t in range(self.num_variables + 1):
                    temp = self.constraint_coefficients[leaving_variable_index][t] * multiplier
                    self.constraint_coefficients[k][t] -= temp

    def print_solution(self):
        if self.solution_found:
            basic_solution = [0] * self.num_variables
            count = 0
            for i in self.basic_variables:
                basic_solution[i] = self.constraint_coefficients[count][-1]
                count += 1
            objective_value = sum(self.objective_coefficients[i] * basic_solution[i] for i in range(self.num_variables))
            print("唯一最优解:", basic_solution, format(objective_value, '.2f'))
        else:
            print("无可行解或目标函数无上界")

# 示例问题
objective_coefficients = [-1, -2, -1, 1, 0]
constraint_coefficients = np.array([
    [-2, 1, -3, 0, 1, -4],
    [-2, -3, -4, 0, 0, 0]
])

solver = DualSimplex(objective_coefficients, constraint_coefficients)
solver.solve()
