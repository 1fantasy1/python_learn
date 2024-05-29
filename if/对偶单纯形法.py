import numpy as np


class DualSimplex(object):
    # 构7函数（初始化函数）
    def __init__(self, z, B, d):
        self.X_count = len(z)  # 变量个数
        self.b_count = len(d)  # 约束条件个数
        self.z = z  # 目标函数
        self.C = []  # 检验数
        self.B = B  # 基变量,由于运算规则必须按顺序给出基变量
        self.d = d  # 约束条件,包括右端常数
        self.flag = 1

    # check()，检验是否为最优解
    def check(self):
        self.flag = 1
        for i in range(self.b_count):
            if self.d[i][self.X_count] < 0:  # 若有约束条件右端常数为负，继续迭代
                self.flag = 0
                break

    # solve()
    def solve(self):
        lim = 100  # 防止无限迭代
        while (lim > 0):
            self.C = []  # 检验数清空
            for j in range(self.X_count):
                zj = 0
                for i in range(self.b_count):  # 遍历第j列全行系数，计算第j个变量检验数
                    zj += self.d[i][j] * self.z[self.B[i]]  # 限制B基变量序号顺序之处
                self.C.append(self.z[j] - zj)  # 检验数,'cj-zj'
            self.check()
            if self.flag > 0:
                break
            else:
                self.pivot()  # 进行基变换（旋转）
                lim -= 1

        self.print()

    # pivot(),基变换（旋转）
    def pivot(self):
        if np.min(d[:][:-1]) >= 0:
            self.special = True
        bi = []
        for i in range(self.b_count):
            bi.append(self.d[i][self.X_count])
        out_num = bi.index(min(bi))  # 确定出基
        Sita = []  # θ
        for j in range(self.X_count):
            if self.d[out_num][j] >= 0 or self.C[j] == 0:  # 被除数不为0，除数必须小于0
                sita = float('inf')  # 给一个正无穷的数，便于排除掉
            else:
                sita = self.C[j] / self.d[out_num][j]
            Sita.append(sita)
        in_num = Sita.index(min(Sita))  # 确定入基
        main = self.d[out_num][in_num]

        self.B[out_num] = in_num  # 更新基变量
        for x in range(self.X_count + 1):  # 变换基变量所在行
            self.d[out_num][x] = self.d[out_num][x] / main
        for k in range(self.b_count):  # 变换其他行
            if k != out_num:
                times = self.d[k][in_num]  # 倍数
                for t in range(self.X_count + 1):
                    temp = self.d[out_num][t] * times
                    self.d[k][t] = self.d[k][t] - temp

    def print(self):
        # 如有最优解输出最优解和目标函数极值
        X = [0] * self.X_count
        count = 0
        for i in self.B:
            X[i] = self.d[count][self.X_count]
            count += 1
        Z = 0
        for i in range(self.X_count):
            Z += self.z[i] * X[i]
        print("有唯一最优解", X, format(Z, '.2f'))


dstr = '''-1 -2 -1 1 0 -3
-2 1 -3 0 1 -4
-2 -3 -4 0 0 0'''
d = np.array([[eval(dij) for dij in dj.split(' ')] for dj in dstr.splitlines()]).astype(float)
(bn, cn) = d.shape  # bn矩阵的行数，cn矩阵的列数
B = list(range(cn - bn, cn - 1))  # 初始基变量的下表组成的列表
m = DualSimplex(d[-1][:-1], B, d[:-1])
n = m.solve()