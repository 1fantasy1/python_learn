"""矩阵链相乘"""
def matrix_chain_order(p):
    n = len(p) - 1  # 矩阵的数量
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(2, n + 1):  # length是链的长度
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m, s

def print_optimal_parens(s, i, j):
    if i == j:
        print(f"A{i}", end="")
    else:
        print("(", end="")
        print_optimal_parens(s, i, s[i][j])
        print_optimal_parens(s, s[i][j] + 1, j)
        print(")", end="")

# 示例使用
p = [30, 35, 15, 5, 10, 20, 25]  # 矩阵链的维度
m, s = matrix_chain_order(p)
print("最小计算代价: ", m[1][len(p) - 1])
print("最优加括号方式: ", end="")
print_optimal_parens(s, 1, len(p) - 1)
print()
