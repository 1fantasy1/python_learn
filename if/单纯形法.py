import numpy as np


def simplex(c, A, b, maximize=False):
    """
    使用单纯形法解决线性规划问题。

    参数:
    c (list): 目标函数的系数向量。
    A (list of lists): 不等式约束条件的系数矩阵。
    b (list): 不等式约束条件的右端向量。
    maximize (bool): 是否为极大化问题。如果为True，则求解极大化问题；否则，求解极小化问题。

    返回:
    dict: 包含求解状态、目标函数最优值、决策变量最优解和松弛变量的字典。
    """
    # 将目标函数系数向量转换为NumPy数组
    if maximize:
        # 如果是极大化问题，将目标函数的系数取负，以转化为极小化问题
        c = -np.array(c)
    else:
        c = np.array(c)

    # 将约束矩阵和右端向量转换为NumPy数组
    A = np.array(A)
    b = np.array(b)

    # 添加松弛变量，将原来的A矩阵扩展
    num_vars = len(c)  # 原始变量的数量
    num_constraints = len(b)  # 约束的数量
    A = np.hstack((A, np.eye(num_constraints)))  # 在A矩阵右侧添加单位矩阵（松弛变量）
    c = np.concatenate((c, np.zeros(num_constraints)))  # 在c向量后面添加0（松弛变量的系数）

    # 初始基可行解
    basis = list(range(num_vars, num_vars + num_constraints))  # 基变量索引（松弛变量）
    non_basis = list(range(num_vars))  # 非基变量索引（原始变量）

    B = A[:, basis]  # 基矩阵B
    N = A[:, non_basis]  # 非基矩阵N
    cb = c[basis]  # 基变量的目标函数系数
    cn = c[non_basis]  # 非基变量的目标函数系数

    # 迭代求解
    while True:
        # 计算当前的基可行解xB
        xB = np.linalg.solve(B, b)

        # 计算检验数
        pi = np.linalg.solve(B.T, cb)
        reduced_costs = cn - pi @ N

        # 检查是否所有检验数都非负
        if np.all(reduced_costs >= 0):
            # 如果所有检验数都非负，找到最优解
            solution = np.zeros_like(c)
            solution[basis] = xB  # 基变量的值
            objective_value = cb @ xB  # 计算目标函数的值
            if maximize:
                objective_value = -objective_value  # 如果是极大化问题，将目标函数值取负还原
            return {
                '状态': '找到最优解',
                '目标函数最优值': objective_value,
                '决策变量最优解': {f'x{i + 1}': solution[i] for i in range(num_vars)},
                '松弛变量': {f's{i + 1}': solution[num_vars + i] for i in range(num_constraints)}
            }

        # 确定进入基的变量（reduced_costs最小的索引）
        entering = np.argmin(reduced_costs)

        # 确定离开基的变量
        d = np.linalg.solve(B, N[:, entering])
        if np.all(d <= 0):
            return {'状态': '问题无界'}  # 如果d中所有元素都<=0，问题无界

        ratios = xB / d
        leaving = np.argmin(np.where(d > 0, ratios, np.inf))  # 计算换出基变量

        # 更新基和非基
        basis[leaving], non_basis[entering] = non_basis[entering], basis[leaving]
        B = A[:, basis]  # 更新基矩阵B
        N = A[:, non_basis]  # 更新非基矩阵N
        cb = c[basis]  # 更新基变量的目标函数系数
        cn = c[non_basis]  # 更新非基变量的目标函数系数

# 书本P30例子
# 目标函数的系数向量
c = [2, -2]

# 不等式约束条件的系数矩阵
A = [[1, 1],
     [-1, 1],
     [6, 2]]

# 不等式约束条件的右端向量
b = [5, 6, 21]

# result = simplex(c, A, b, maximize=True)
# print("极大化问题结果:")
# for key, value in result.items():
#     print(f"{key}: {value}")

result = simplex(c, A, b, maximize=False)
print("\n极小化问题结果:")
for key, value in result.items():
    print(f"{key}: {value}")
