# def f(x):
#     # 定义目标函数 f(x) = x^4 - x^2 - 2x - 5
#     return x ** 4 - x ** 2 - 2 * x - 5
#
#
# def success_failure_method_interval(x0, h0, tolerance=1e-6, max_iter=1000):
#     """
#     使用成功失败法寻找单峰函数最小值所在的区间
#
#     参数:
#     x0 - 初始值
#     h0 - 初始步长
#     tolerance - 容差，用于控制迭代的停止条件
#     max_iter - 最大迭代次数
#
#     返回值:
#     interval - 包含最小值的区间 [left_bound, right_bound]
#     f_best - 近似最小值
#     """
#     # 初始化当前最优解 x_best 和对应的函数值 f_best
#     x_best = x0
#     f_best = f(x_best)
#     h = h0  # 初始化步长
#
#     left_bound = x_best
#     right_bound = x_best
#
#     for _ in range(max_iter):
#         # 计算当前点左右的函数值
#         f_right = f(x_best + h)
#         f_left = f(x_best - h)
#
#         if f_right < f_best:
#             # 如果右边点的函数值更小，更新 x_best 和 f_best，并更新右边界
#             x_best += h
#             f_best = f_right
#             right_bound = x_best
#         elif f_left < f_best:
#             # 如果左边点的函数值更小，更新 x_best 和 f_best，并更新左边界
#             x_best -= h
#             f_best = f_left
#             left_bound = x_best
#         else:
#             # 如果两边都没有更小的函数值，减小步长 h
#             h *= 0.5
#
#         # 如果步长小于容差，停止迭代
#         if h < tolerance:
#             break
#
#     interval = (left_bound, right_bound)
#     return interval, f_best
#
#
# # 初始化值
# x0 = 0
# h0 = 0.01
#
# # 运行成功失败法
# interval, f_min = success_failure_method_interval(x0, h0)
#
# print(interval,f_min)
def f(x):
    return x**4 - x**2 - 2*x - 5

# 初始化参数
x0 = 0
h0 = 0.01

# 成功失败法
while True:
    if f(x0 + h0) < f(x0):
        x0 += h0
    else:
        x0 -= h0
    # 检查是否找到了最小值区间
    if f(x0 + h0) >= f(x0) and f(x0 - h0) >= f(x0):
        break

# 最小值区间
min_value_interval = (x0 - h0, x0 + h0)
print(min_value_interval)
