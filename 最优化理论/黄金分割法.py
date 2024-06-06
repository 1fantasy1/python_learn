import numpy as np

# 定义目标函数
def f(x):
    return np.exp(-x) + x**2

# 黄金分割法求解最小值
def golden_section_search(f, a, b, tol=0.2):
    gr = (np.sqrt(5) - 1) / 2  # 黄金比例

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while (b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # 重新计算新的c和d
        c = b - gr * (b - a)
        d = a + gr * (b - a)

    # 返回最小值的区间
    return (a, b)

# 初始区间[0, 1]
a = 0
b = 1

# 终止条件
tolerance = 0.2

# 进行黄金分割法搜索
result = golden_section_search(f, a, b, tolerance)

print(f"最小值所在的区间: [{result[0]}, {result[1]}]")
print(f"在此区间内函数值分别为: f({result[0]}) = {f(result[0])}, f({result[1]}) = {f(result[1])}")