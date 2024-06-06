# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义函数 f(x)
# def f(x):
#     return -x**2 * np.exp(-x**2)
#
# # 成功失败法
# def success_failure_method(f, x0, h0, max_iter=1000, tol=1e-6):
#     x = x0
#     h = h0
#     for _ in range(max_iter):
#         if f(x + h) < f(x):
#             x += h
#             h *= 2
#         else:
#             h = -h / 2
#         if abs(h) < tol:
#             break
#     return x, x + h
#
# # 初始值
# x0 = 2
# h0 = 0.01
#
# # 找到单峰区间
# interval = success_failure_method(f, x0, h0)
# print("单峰区间:", interval)
#
# # 绘制函数图像
# x = np.linspace(-3, 3, 400)
# y = f(x)
#
# plt.figure(figsize=(8, 6))
# plt.plot(x, y, label=r'$f(x)=-x^2 e^{-x^2}$')
# plt.axvline(interval[0], color='r', linestyle='--', label=f'Unimodal Interval Start: {interval[0]:.6f}')
# plt.axvline(interval[1], color='g', linestyle='--', label=f'Unimodal Interval End: {interval[1]:.6f}')
# plt.title(r'Plot of $f(x)=-x^2 e^{-x^2}$ over [-3, 3]')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
def f(x):
    return -x**2 * np.exp(-x**2) #32264700042 杨金城

# 初始化参数
x0 = 2
h0 = 0.01

# 成功失败法
iteration = 0  # 记录迭代次数
print(f"迭代第{iteration}次: x0 = {x0}, f(x0) = {f(x0)}") #32264700042 杨金城
while True:
    iteration += 1
    if f(x0 + h0) < f(x0):
        x0 += h0
    else:
        x0 -= h0
    print(f"迭代第{iteration}次: x0 = {x0}, f(x0) = {f(x0)}") #32264700042 杨金城
    # 检查是否找到了最小值区间
    if f(x0 + h0) >= f(x0) and f(x0 - h0) >= f(x0):
        break

# 最小值区间
min_value_interval = (x0 - h0, x0 + h0)
print(f"最小值所在区间: {min_value_interval}")
