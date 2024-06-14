import numpy as np


# 定义目标函数
def f(x):
    return (x[0] - 1) ** 4 + x[1] ** 2


# 定义目标函数的梯度
def grad_f(x):
    return np.array([4 * (x[0] - 1) ** 3, 2 * x[1]])


# 定义目标函数的Hessian矩阵
def hessian_f(x):
    return np.array([[12 * (x[0] - 1) ** 2, 0], [0, 2]])


# 牛顿法
def newton_method(x0, epsilon=1e-8, max_iterations=100000):
    x = x0
    for i in range(max_iterations):
        gradient = grad_f(x)
        hessian = hessian_f(x)
        norm_gradient = np.linalg.norm(gradient)

        # 检查是否收敛
        if norm_gradient < epsilon:
            break

        # 计算牛顿步长
        delta_x = np.linalg.solve(hessian, -gradient)

        # 更新点
        x = x + delta_x

        # 打印迭代信息
        print(f"迭代第{i + 1}次: x = {x}, f(x) = {f(x)}, 梯度范数 = {norm_gradient}")

    return x


# 初始点
x0 = np.array([0, 1])

# 运行牛顿法
solution = newton_method(x0)

print(f"优化结果: x = {solution}, f(x) = {f(solution)}")