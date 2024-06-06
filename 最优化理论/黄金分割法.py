import numpy as np

# 定义目标函数
def f(x): #32264700042 杨金城
    return np.exp(-x) + x**2

# 黄金分割法求解最小值
def golden_section_search(f, a, b, tol=0.2):
    gr = (np.sqrt(5) - 1) / 2  # 黄金比例
    iteration = 0

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while (b - a) > tol: #32264700042 杨金城
        iteration += 1
        if f(c) < f(d):
            b = d
        else:
            a = c

        # 重新计算新的c和d
        c = b - gr * (b - a)
        d = a + gr * (b - a)

        # 输出迭代过程
        print(f"迭代第{iteration}次: 区间 = [{a}, {b}], c = {c}, d = {d}, f(c) = {f(c)}, f(d) = {f(d)}")

    # 返回最小值的区间
    return (a, b)

# 初始区间[0, 1]
a = 0
b = 1

# 终止条件，要求最终区间间隔小于0.2
tolerance = 0.2

# 进行黄金分割法搜索
result = golden_section_search(f, a, b, tolerance) #32264700042 杨金城

print(f"最小值所在的区间: [{result[0]}, {result[1]}]")
print(f"在此区间内函数值分别为: f({result[0]}) = {f(result[0])}, f({result[1]}) = {f(result[1])}")