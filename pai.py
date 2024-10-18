from mpmath import mp
from tqdm import tqdm

# 设置精度
mp.dps = 1000


# 使用Chudnovsky算法计算圆周率
def chudnovsky(num_iterations):
    pi = mp.mpf(0)
    chunk_size = 1000  # 每次计算的块大小
    num_chunks = num_iterations // chunk_size

    for chunk in tqdm(range(num_chunks)):
        pi_chunk = mp.mpf(0)
        # 遍历本块的所有项
        for k in range(chunk * chunk_size, (chunk + 1) * chunk_size):
            # 计算分子，使用(-1)^k和阶乘函数
            numerator = (-1) ** k * mp.fac(6 * k) * (13591409 + 545140134 * k)
            # 计算分母，同样使用阶乘函数
            denominator = mp.fac(3 * k) * (mp.fac(k) ** 3) * (640320 ** (3 * k))
            # 累加本项对圆周率的贡献
            pi_chunk += mp.mpf(numerator) / mp.mpf(denominator)
        # 根据本块的计算结果调整pi_chunk的值
        pi_chunk = pi_chunk * mp.mpf(12) / mp.mpf(640320 ** 1.5)
        # 反转pi_chunk以修正计算误差
        pi_chunk = 1 / pi_chunk
        # 累加到全局圆周率估计上
        pi += pi_chunk


    return pi


# 设置迭代次数
num_iterations = 10000  # 调整此值以增加计算精度

# 计算圆周率并显示进度条
pi = chudnovsky(num_iterations)

# 打印结果
print("Pi:", pi)