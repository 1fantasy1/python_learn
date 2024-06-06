import pandas as pd
import numpy as np


def dual_simplex_method(c, A, b, maximize=False): #32264700042 杨金城
    """
    使用对偶单纯形法求解线性规划问题。

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
                        Simplex_table.loc[x] = Simplex_table.loc[x] - Simplex_table.loc[x_out] * Simplex_table.loc[
                            x, x_in]
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
    if solver.Simplex_table is not None: #32264700042 杨金城
        # 提取最优解和最优目标函数值
        optimal_solution = solver.Simplex_table.iloc[:-1, 0]
        optimal_value = solver.Simplex_table.iloc[-1, 0]
        # 如果是极大化问题，需要将结果转回正值
        if maximize:
            optimal_value = -optimal_value
        return optimal_solution, optimal_value
    else:
        return None, None


# 测试脚本
if __name__ == '__main__': #32264700042 杨金城
    # 定义参数
    A = [[1, 2, 1],
         [2, -1, 3]]
    b = [3, 4]
    c = [2, 3, 4]

    # 调用函数求解极小化问题
    optimal_solution, optimal_value = dual_simplex_method(c, A, b, maximize=False) #32264700042 杨金城
    if optimal_solution is not None:
        print(f"极小化问题的最优解为\n{optimal_solution}")
        print(f'极小化问题的最优值为{optimal_value}')
    else:
        print("该极小化问题无解或无法解决")

    # # 调用函数求解极大化问题
    # optimal_solution, optimal_value = dual_simplex_method(c, A, b, maximize=True)
    # 最优化理论 optimal_solution is not None:
    #     print(f"极大化问题的最优解为\n{optimal_solution}")
    #     print(f'极大化问题的最优值为{optimal_value}')
    # else:
    #     print("该极大化问题无解或无法解决")
