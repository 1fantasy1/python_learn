import numpy as np
from math import pi
print("debug_begin");
np.random.seed(142857)
def gen_spiral(size=50, n=4, scale=2):
    xs = np.zeros((size * n, 2), dtype=np.float32)
    ys = np.zeros(size * n, dtype=np.int8)
    for i in range(n):
        ix = range(size * i, size * (i + 1))
        r = np.linspace(0.0, 1, size+1)[1:]
        t = np.linspace(2 * i * pi / n, 2 * (i + scale) * pi / n, size) + np.random.random(size=size) * 0.1
        xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        ys[ix] = 2 * (i % 2) - 1
    return xs, ys
print("debug_end");
xs, ys = gen_spiral()
def rbf_kernel(x1, x2, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

ys = np.where(ys == 0, -1, 1)

num_samples = xs.shape[0]
alphas = np.zeros(num_samples)

num_iterations = 10

for epoch in range(num_iterations):
    for i in range(num_samples):
        prediction = 0
        for j in range(num_samples):
            if alphas[j] != 0:
                prediction += alphas[j] * ys[j] * rbf_kernel(xs[j], xs[i])
        pred = np.sign(prediction)
        if pred != ys[i]:
            alphas[i] += 1

correct = 0
for i in range(num_samples):
    prediction = 0
    for j in range(num_samples):
        if alphas[j] != 0:
            prediction += alphas[j] * ys[j] * rbf_kernel(xs[j], xs[i])
    pred = np.sign(prediction)
    if pred == ys[i]:
        correct += 1

acc = correct / num_samples

def test_acc(acc):
    res = True if acc>0.85 else False
    print(res)

test_acc(acc)