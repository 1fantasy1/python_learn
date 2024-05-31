import numpy as np
import pandas as pd
from colorama import init, Fore, Back, Style
def Simplex_method(c, A, b, maximize=False): #32264700042 杨金城
    """
        解决线性规划问题的函数。

        参数:
        c (list): 目标函数的系数列表。
        A (list of lists): 约束条件矩阵。
        b (list): 约束条件的右侧常数项列表。
        maximize (bool): 指示是否为极大化问题。默认为 False（即极小化问题）。

        返回:
        dict: 包含求解状态、目标函数最优值、决策变量最优解和松弛变量的字典。
    """
    def print_simplex_table(iteration, basis, non_basis, B, N, cb, cn, xB, reduced_costs): #32264700042 杨金城
        """
               打印单纯形表。

               参数:
               iteration (int): 当前迭代次数。
               basis (list): 基变量索引列表。
               non_basis (list): 非基变量索引列表。
               B (numpy.ndarray): 基变量系数矩阵。
               N (numpy.ndarray): 非基变量系数矩阵。
               cb (numpy.ndarray): 基变量目标函数系数。
               cn (numpy.ndarray): 非基变量目标函数系数。
               xB (numpy.ndarray): 基变量解值。
               reduced_costs (numpy.ndarray): 检验数。
        """
        print(f"迭代 {iteration}:")

        # 打印基变量和非基变量的索引及目标函数系数
        basis_vars = [f"x{i + 1}" for i in basis]
        non_basis_vars = [f"x{i + 1}" for i in non_basis]
        all_vars = basis_vars + non_basis_vars
        all_coeffs = np.concatenate((cb, cn))

        # 创建综合矩阵
        combined_matrix = np.hstack((B, N))

        # 打印表头
        print(f"{'变量':>8}", end=" ")
        for var in all_vars:
            print(f"{var:>8}", end=" ")
        print(f"{'检验数':>8}")

        # 打印系数矩阵和检验数
        for i in range(combined_matrix.shape[0]):
            print(f"{basis_vars[i] if i < len(basis_vars) else '':>8}", end=" ")
            for elem in combined_matrix[i]:
                print(f"{elem:>8.2f}", end=" ")
            if i < len(basis_vars):
                print(f"{xB[i]:>8.2f}", end=" ")
            print()

        # 打印目标函数系数行
        print(f"{'c':>8}", end=" ")
        for coeff in all_coeffs:
            print(f"{coeff:>8.2f}", end=" ")
        print()

        # 打印检验数
        print(f"{'检验数':>8}", end=" ")
        for rc in reduced_costs:
            print(f"{rc:>8.2f}", end=" ")
        print()

        # 打印当前解
        print(f"{'当前解':>8} ", end="")
        for xb in xB:
            print(f"{xb:>8.2f}", end=" ")
        print()
        print("-" * 50)
    def simplex(c, A, b, maximize=False): #32264700042 杨金城
        if maximize:
            c = -np.array(c)
        else:
            c = np.array(c)

        A = np.array(A)
        b = np.array(b)

        num_vars = len(c)
        num_constraints = len(b)
        A = np.hstack((A, np.eye(num_constraints)))
        c = np.concatenate((c, np.zeros(num_constraints)))

        basis = list(range(num_vars, num_vars + num_constraints))
        non_basis = list(range(num_vars))

        B = A[:, basis]
        N = A[:, non_basis]
        cb = c[basis]
        cn = c[non_basis]

        iteration = 0
        while True:
            iteration += 1

            xB = np.linalg.solve(B, b)

            pi = np.linalg.solve(B.T, cb)
            reduced_costs = cn - pi @ N

            print_simplex_table(iteration, basis, non_basis, B, N, cb, cn, xB, reduced_costs)

            if np.all(reduced_costs >= 0):
                solution = np.zeros_like(c)
                solution[basis] = xB
                objective_value = cb @ xB
                if maximize:
                    objective_value = -objective_value
                return {
                    '状态': '找到最优解',
                    '目标函数最优值': objective_value,
                    '决策变量最优解': {f'x{i + 1}': solution[i] for i in range(num_vars)},
                    '松弛变量': {f's{i + 1}': solution[num_vars + i] for i in range(num_constraints)}
                }

            entering = np.argmin(reduced_costs)

            d = np.linalg.solve(B, N[:, entering])
            if np.all(d <= 0):
                return {'状态': '问题无界'}

            ratios = xB / d
            leaving = np.argmin(np.where(d > 0, ratios, np.inf))

            basis[leaving], non_basis[entering] = non_basis[entering], basis[leaving]
            B = A[:, basis]
            N = A[:, non_basis]
            cb = c[basis]
            cn = c[non_basis]

    # 调用 simplex 函数
    result = simplex(c, A, b, maximize=maximize)

    # 打印结果
    if maximize:
        print("\n极大化问题结果:")
    else:
        print("\n极小化问题结果:")
    for key, value in result.items():
        print(f"{key}: {value}")

def dual_simplex_method(c, A, b, maximize=False): #32264700042 杨金城
    """
    使用单纯形法或对偶单纯形法求解线性规划问题。

    参数:
    c (list of floats): 目标函数系数向量，一维列表，每个元素对应一个变量在目标函数中的系数。
    A (list of list of floats): 约束系数矩阵，二维列表，每行对应一个约束，每列对应一个变量的系数。
    b (list of floats): 资源系数向量，一维列表，每个元素对应一个约束的右端常数项。
    maximize (bool): 是否为极大化问题，默认为False，即极小化问题。

    返回:
    optimal_solution (pandas.Series): 最优解向量，对应于每个变量的最优值。
    optimal_value (float): 最优目标函数值。
    """

    class DualSimplex: #32264700042 杨金城
        def __init__(self, A0, b0, ci):
            self.A0 = A0  # 约束系数矩阵
            self.b0 = b0  # 资源系数向量
            self.ci = ci  # 目标函数系数向量
            self.index_nums = len(self.A0)  # 约束数量
            self.columns_nums = len(self.A0[0])  # 变量数量
            self.ci_up = self.get_ci_up()  # 获取扩展后的目标函数系数向量
            self.data = self.form_data()  # 构建初始单纯形表
            self.Simplex_table = pd.DataFrame(
                data=self.update_data(),  # 更新后的单纯形表数据
                index=self.get_index(),  # 行索引
                columns=self.get_columns()  # 列索引
            )
            self.Simplex_table = self.doo(self.Simplex_table)  # 进行对偶单纯形法计算

        def get_ci_up(self):
            """
            扩展目标函数系数向量，将0插入到第一个位置
            """
            self.ci.insert(0, 0)
            return self.ci

        def form_data(self):
            """
            构建初始单纯形表数据，在每行的第一个位置插入资源系数向量的对应元素
            """
            bi = 0
            for row in self.A0:
                row.insert(0, self.b0[bi])
                bi += 1
            self.A0.append(self.ci_up)
            return self.A0

        def get_index(self):
            """
            获取行索引，行索引包括变量和目标函数行
            """
            index = [f'x{j + 1 + (self.columns_nums - self.index_nums)}' for j in range(self.index_nums)]
            index.append('θ')
            return index

        def get_columns(self):
            """
            获取列索引，列索引包括资源系数列和变量列
            """
            columns = [f'x{i + 1}' for i in range(self.columns_nums)]
            columns.insert(0, 'b')
            return columns

        def update_data(self):
            """
            更新初始单纯形表数据，将其变为负数
            """
            data_update = -1 * np.array(self.data, dtype=float)
            return data_update

        def doo(self, Simplex_table):
            """
            对偶单纯形法主过程，更新单纯形表直到找到最优解或确定无解
            """
            k = 0  # 迭代计数器
            print('初始对偶单纯形表为:')
            print(Simplex_table)
            b_min = Simplex_table.iloc[:self.index_nums, 0].min()
            while b_min < 0:
                # 找到b列中最小的负数对应的行，作为离开基变量
                x_out = Simplex_table.iloc[:self.index_nums, 0].idxmin()
                cta = []
                series_index = []
                # 计算进入基变量
                for i in range(1, self.columns_nums + 1):
                    if Simplex_table.loc[x_out, f'x{i}'] < 0:
                        cta.append(Simplex_table.loc['θ', f'x{i}'] / Simplex_table.loc[x_out, f'x{i}'])
                        series_index.append(f'x{i}')
                x_in_index = cta.index(min(cta))
                x_in = series_index[x_in_index]
                # 更新离开基变量行
                Simplex_table.loc[x_out] = (Simplex_table.loc[x_out]) / Simplex_table.loc[x_out, x_in]
                # 更新其他行
                for x in (Simplex_table.index).tolist():
                    if x != x_out:
                        Simplex_table.loc[x] = Simplex_table.loc[x] - Simplex_table.loc[x_out] * Simplex_table.loc[x, x_in]
                # 更新行索引
                out_x_list = Simplex_table.index.tolist()
                out_x_index = out_x_list.index(x_out)
                out_x_list[out_x_index] = x_in
                Simplex_table.index = out_x_list
                k += 1
                b_min = Simplex_table.iloc[:self.index_nums, 0].min()
                print(f'第{k}次单纯形表为:')
                print(Simplex_table)
                if k == 999:
                    print("该问题无解或无法解决")
                    return None
            return Simplex_table

    # 如果是极大化问题，转换为极小化问题
    if maximize:
        c = [-ci for ci in c]

    # 创建对偶单纯形法实例并计算最优解
    solver = DualSimplex(A, b, c)
    if solver.Simplex_table is not None:
        # 提取最优解和最优目标函数值
        optimal_solution = solver.Simplex_table.iloc[:-1, 0]
        optimal_value = solver.Simplex_table.iloc[-1, 0]
        # 如果是极大化问题，需要将结果转回正值
        if maximize:
            optimal_value = -optimal_value
        return optimal_solution, optimal_value
    else:
        return None, None
def General_simplex_method(c, A, b, maximize,WTUDM):
    if WTUDM is True:
        if maximize is False:
            # 调用函数求解极小化问题
            optimal_solution, optimal_value = dual_simplex_method(c, A, b, maximize)  # 调用对偶单纯形法
            if optimal_solution is not None:
                print(f"极小化问题的最优解为\n{optimal_solution}")
                print(f'极小化问题的最优值为{optimal_value}')
            else:
                print("该极小化问题无解或无法解决")
        else:
            # 调用函数求解极大化问题
            optimal_solution, optimal_value = dual_simplex_method(c, A, b, maximize=True)
            if optimal_solution is not None:
                print(f"极大化问题的最优解为\n{optimal_solution}")
                print(f'极大化问题的最优值为{optimal_value}')
            else:
                print("该极大化问题无解或无法解决")

    else:
        Simplex_method(c, A, b, maximize)  # 调用普通单纯形法
if __name__ == '__main__': #32264700042 杨金城
    init()
    print("调用普通单纯形法")
    A = [[1, 1],
        [-1, 1],
        [6, 2]]
    b = [5, 6, 21]
    c = [2, -2]
    # 调用单纯形法的函数
    General_simplex_method(c, A, b, maximize=False,WTUDM=False)
    print(Fore.RED + "#" * 80)
    print(Style.RESET_ALL + "调用对偶单纯形法")
    # 定义参数
    A = [[1, 2, 1],
         [2, -1, 3]]
    b = [3, 4]
    c = [2, 3, 4]

    General_simplex_method(c, A, b, maximize=False,WTUDM=True)
    # General_simplex_method(c, A, b, maximize=True)