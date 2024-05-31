# import numpy as np
# from scipy.optimize import linprog
#
# def solve_linear_program(c, A, b, maximize=False):
#     """
#     解决线性规划问题。
#
#     参数:
#     c (list): 目标函数的系数。
#     A (list of lists): 不等式约束的系数矩阵。
#     b (list): 不等式约束的右侧向量。
#     maximize (bool): 如果为 True，解决极大化问题。如果为 False，解决极小化问题。
#
#     返回:
#     dict: 包含最优值和解向量的字典，如果优化失败，则包含错误信息。
#     """
#     # 如果我们要求极大化问题，需要将目标函数的系数取反
#     if maximize:
#         c = [-ci for ci in c]
#
#     # 确定变量的数量
#     num_vars = len(c)
#
#     # 设置每个变量的边界（默认非负）
#     bounds = [(0, None)] * num_vars
#
#     # 解决线性规划问题
#     res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
#
#     # 输出结果
#     if res.success:
#         # 如果是极大化问题，结果需要取反
#         optimal_value = -res.fun if maximize else res.fun
#         return {'Optimal value': optimal_value, 'X': res.x}
#     else:
#         return {'Error': res.message}
#
# # 示例使用
# # 目标函数的系数
# c = [2, 3, 4]
#
# # 不等式约束的左侧矩阵
# A = [[-1, -2, -1],
#      [-2, 1, -3]]
#
# # 不等式约束的右侧向量
# b = [-3, -4]
#
# # # 解决极大化问题
# # result = solve_linear_program(c, A, b, maximize=True)
# # print(result)
#
# # 解决极小化问题
# result = solve_linear_program(c, A, b, maximize=False)
# print(result)
#
# # c = np.array([2, 3, 4])
# # A = np.array([[1, 2, 1],
# #               [2, -1, 3]])
# # b = np.array([3, 4])
import pandas as pd, numpy as np


# 求最小问题的对偶单纯形法
class Dual_Simplex_Method_minimize(object):
    # 初始化表，将已经标准化的原问题的A0,b0，输入进来
    def __init__(self, A0, b0, ci):
        # 初始化
        self.A0 = A0
        self.b0 = b0
        self.ci = ci
        # 获取行列数
        self.index_nums = len(self.A0)  # 行数
        self.columns_nums = len(self.A0[0])  # 列数
        # 获取增广矩阵
        self.ci_up = self.get_ci_up()
        self.data = self.form_data()  # 矩阵形式的data,初始数据
        # 初始化表，将原表取负后的表
        self.Simplex_table = pd.DataFrame(
            data=self.update_data(),  # 取负后更新过的数据
            index=self.get_index(),
            columns=self.get_columns()
        )
        self.Simplex_table = self.doo(self.Simplex_table)

    # 获取ci的增广矩阵
    def get_ci_up(self):
        self.ci.insert(0, 0)
        return self.ci

    # 矩阵的拼接，形成data数据表
    def form_data(self):
        bi = 0
        for list in self.A0:
            list.insert(0, self.b0[bi])
            bi += 1
        self.A0.append(self.ci_up)
        return self.A0

    # 获取单纯形表的行标签
    def get_index(self):
        index = [f'x{j + 1 + (self.columns_nums - self.index_nums)}' for j in range(self.index_nums)]
        index.append('cgm')
        return index

    # 获取单纯形表的列标签
    def get_columns(self):
        columns = [f'x{i + 1}' for i in range(self.columns_nums)]
        columns.insert(0, 'b')
        return columns

    # 最大值转最小值，更新单纯形表数据
    def update_data(self):
        data_update = -1 * np.array(self.data, dtype=float)
        return data_update

    # 定义推进函数
    def doo(self, Simplex_table):
        k = 0
        print('初始对偶单纯形表为:')
        print(Simplex_table)
        # 判断b是否小于0，否则，则为最优解
        b_min = Simplex_table.iloc[:self.index_nums, 0].min()
        while b_min < 0:
            # 确定出基变量
            x_out = Simplex_table.iloc[:self.index_nums, 0].idxmin()  # 最小的b出基
            # 确定进基变量
            cta = []
            series_index = []
            for i in range(1, self.columns_nums + 1):
                if Simplex_table.loc[x_out, f'x{i}'] < 0:
                    cta.append(Simplex_table.loc['cgm', f'x{i}'] / Simplex_table.loc[x_out, f'x{i}'])
                    series_index.append(f'x{i}')
            x_in_index = cta.index(min(cta))
            x_in = series_index[x_in_index]
            # 行初等变化
            # 出基行除以对应位化为1
            Simplex_table.loc[x_out] = (Simplex_table.loc[x_out]) / Simplex_table.loc[x_out, x_in]
            for x in (Simplex_table.index).tolist():
                if x != x_out:
                    Simplex_table.loc[x] = Simplex_table.loc[x] - Simplex_table.loc[x_out] * \
                                           Simplex_table.loc[x, x_in]
            # 交换出基进基变量标签，即更新表格行标签，将进基变量和出基变量的位置对调
            out_x_list = Simplex_table.index.tolist()
            out_x_index = out_x_list.index(x_out)
            out_x_list[out_x_index] = x_in
            Simplex_table.index = out_x_list
            k += 1
            b_min = Simplex_table.iloc[:self.index_nums, 0].min()
            print(f'第{k}次单纯形表为:')
            print(Simplex_table)
            if i == 999:
                print("该问题无解或无法解决")
                return 0
        return Simplex_table


# 脚本自调试
if __name__ == '__main__':
    # 写入化为标准型后的约束系数矩阵，资源系数矩阵和价值系数矩阵（填最开始的ci）
    A = [[1,2,1],
         [2,-1,3]]
    b = [3,4]
    c = [2,3,4]
    # 实例化对象
    test = Dual_Simplex_Method_minimize(A, b, c)
    print(f"最优解为\n{test.Simplex_table.iloc[:-1, 0]}")
    print(f'最优化结果为{test.Simplex_table.iloc[-1, 0]}')