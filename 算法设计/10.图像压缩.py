"""图像压缩"""
import math

def compress(p):
    n = len(p)
    Lmax = 255  # 像素序列最大长度
    header = 11 # 每一段的头部信息 = 11
    s = [0] * (n + 1)
    l = [0] * (n + 1)
    b = [0] * (n + 1)

    for i in range(1, n + 1):
        b[i] = math.ceil(math.log2(p[i - 1] + 1))  # 计算灰度值所需的位数
        bMax = b[i]
        s[i] = s[i - 1] + bMax
        l[i] = 1

        for j in range(2, min(i, Lmax) + 1):
            if bMax < b[i - j + 1]:
                bMax = b[i - j + 1]
            if s[i] > s[i - j] + j * bMax:
                s[i] = s[i - j] + j * bMax
                l[i] = j

        s[i] += header

    segments = []
    while n > 0:
        segments.append(n)
        n -= l[n]
    segments.reverse()

    print("图像压缩后的最小空间为:", s[len(p)])
    print("将原灰度序列分成", len(segments) - 1, "段序列段")
    for i in range(1, len(segments)):
        segment_length = segments[i] - segments[i - 1]
        print(f"段长度: {segment_length}, 所需存储位数: {b[segments[i]]}")

if __name__ == "__main__":
    p = [int(x) for x in input("请输入灰度值序列 (以空格分隔): ").split()]
    print("图像的灰度序列为:", p)
    compress(p)