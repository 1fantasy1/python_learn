import numpy as np


# 定义目标函数
def f(x):
    return 2 * x[0] ** 2 + x[1] ** 2


# 定义目标函数的梯度
def grad_f(x):
    return np.array([4 * x[0], 2 * x[1]])


# 最速下降法
def steepest_descent(x0, epsilon=0.1, max_iterations=1000):
    x = x0
    for i in range(max_iterations):
        gradient = grad_f(x)
        norm_gradient = np.linalg.norm(gradient)

        # 检查是否收敛
        if norm_gradient < epsilon:
            break

        # 确定搜索方向
        d = -gradient

        # 线搜索确定步长 λ
        # λ = argmin f(x + λd)
        # 使用解析方法找到最佳 λ (即 λ = 5/18)
        # 一般情况可以用数值方法找到 λ 但这里用解析解
        lambda_opt = (gradient @ gradient) / (4 * gradient[0] ** 2 + 2 * gradient[1] ** 2)

        # 更新点
        x = x + lambda_opt * d

        # 打印迭代信息
        print(f"迭代第{i + 1}次: x = {x}, f(x) = {f(x)}, 梯度范数 = {norm_gradient}")

    return x


# 初始点
x0 = np.array([1, 1])

# 运行最速下降法
solution = steepest_descent(x0)

print(f"优化结果: x = {solution}, f(x) = {f(solution)}")