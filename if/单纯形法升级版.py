import numpy as np
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
if __name__ == '__main__': #32264700042 杨金城
    # 书本P30例子
    A = [[1, 1],
        [-1, 1],
        [6, 2]]
    b = [5, 6, 21]
    c = [2, -2]
    # 调用单纯形法的函数
    Simplex_method(c, A, b, maximize=False)
