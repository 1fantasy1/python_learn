"""最长公共子序列"""
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 从 dp 表中恢复最长公共子序列
    lcs_length = dp[m][n]
    lcs_sequence = []
    i, j = m, n

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs_sequence.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_sequence.reverse()
    return lcs_length, ''.join(lcs_sequence)

# 示例使用
X = "AGGTAB"
Y = "GXTXAYB"
lcs_length, lcs_sequence = lcs(X, Y)
print(f"最长公共子序列的长度: {lcs_length}")
print(f"最长公共子序列: {lcs_sequence}")
