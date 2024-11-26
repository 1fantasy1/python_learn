def gradient_descent_1d(grad, cur_x, learning_rate, precision, max_iters):
    """
    一维梯度下降法

    :param grad: 目标函数的梯度函数，接收当前x值并返回梯度
    :param cur_x: 当前x值，提供初始值
    :param learning_rate: 学习率，决定步长大小
    :param precision: 收敛精度，当梯度的绝对值小于此值时停止迭代
    :param max_iters: 最大迭代次数
    :return: 局部最小值对应的x的取值
    """
    for _ in range(max_iters):
        gradient = grad(cur_x)
        if abs(gradient) < precision:
            break
        cur_x -= learning_rate * gradient
    return cur_x

# 示例用法
if __name__ == "__main__":
    # 定义目标函数 y = x^2 + 1 的梯度 y' = 2x
    def grad(x):
        return 2 * x

    initial_x = 10.0       # 初始值
    learning_rate = 0.1    # 学习率
    precision = 1e-6       # 收敛精度
    max_iters = 1000       # 最大迭代次数

    minimum_x = gradient_descent_1d(grad, initial_x, learning_rate, precision, max_iters)
    print(f"局部最小值对应的 x 取值为: {minimum_x}")