"""fibonacci数"""
def fibonacci_dynamic(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        fib = [0] * (n + 1)
        fib[1] = 1
        for i in range(2, n + 1):
            fib[i] = fib[i - 1] + fib[i - 2]
        return fib[n]

# 示例使用
n = 10
print(f"斐波那契数列第 {n} 项是 {fibonacci_dynamic(n)}")
