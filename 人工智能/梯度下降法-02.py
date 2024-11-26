import math
import numpy as np

print("debug_begin");
import numpy as np


def func_2d_test1(x):
    return - math.exp(-(x[0] ** 2 + x[1] ** 2))


def grad_2d_test1(x):
    deriv0 = 2 * x[0] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    deriv1 = 2 * x[1] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    return np.array([deriv0, deriv1])


def func_2d_test2(x):
    return x[0] ** 2 + x[1] ** 2 + 2 * x[0] + 1


def grad_2d_test2(x):
    deriv0 = 2 * x[0] + 2
    deriv1 = 2 * x[1]
    return np.array([deriv0, deriv1])


print("debug_end");


def gradient_descent_2d(grad, cur_x=np.array([0.1, 0.1]), learning_rate=0.01, precision=0.0001, max_iters=10000):
    previous_step_size = 1
    iters = 0

    while previous_step_size > precision or np.linalg.norm(grad(cur_x)) > precision:
        if iters >= max_iters:
            break
        prev_x = cur_x.copy()
        gradient = grad(cur_x)
        cur_x = cur_x - learning_rate * gradient
        previous_step_size = np.linalg.norm(cur_x - prev_x)
        iters += 1

    return cur_x


print("debug_begin");
import numpy as np


def test():
    res = gradient_descent_2d(grad_2d_test1, cur_x=np.array([1, -1]), learning_rate=0.2, precision=0.0001,
                              max_iters=10000)
    print("%.7f %.7f" % (res[0], res[1]))
    res2 = gradient_descent_2d(grad_2d_test2, cur_x=np.array([2, 2]), learning_rate=0.2, precision=0.0001,
                               max_iters=10000)
    print("%.7f %.7f" % (res2[0], res2[1]))


print("debug_end");

test()
