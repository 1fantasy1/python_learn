import numpy as np


def danchunXB(c, A, b, BV, tar):
    if (tar == 'max'):
        c = -c
    Ab = np.append(A, b, axis=1)
    Cost = np.append(c, 0)
    temp = np.array([Cost[y] for y in BV]).reshape(1, len(BV))
    zjcj = Cost - np.dot(temp, Ab)
    while np.min(zjcj) < 0:

        temp = np.array([Cost[y] for y in BV]).reshape(1, len(BV))
        zjcj = Cost - np.dot(temp, Ab)
        enter = np.argmin(zjcj[0:])
        # 找换出基变量
        theta = np.zeros(Ab.shape[0])
        for i in range(0, Ab.shape[0]):
            if Ab[i, enter] > 0:
                theta[i] = Ab[i, Ab.shape[1] - 1] / Ab[i, enter]
            else:
                theta[i] = float('inf')
        if (np.min(theta) == float('inf')):
            print('该问题无解')
            break
        leaveindex = np.argmin(theta)
        leave = BV[leaveindex]
        print('换出的变量：', leave + 1)
        print(Ab, '：旋转前Ab')
        BV[leaveindex] = enter
        # 高斯消元
        for i in range(0, Ab.shape[0]):
            if i != leaveindex:
                Ab[i,] = Ab[i,] - Ab[i, enter] * Ab[leaveindex,] / Ab[leaveindex, enter]
            else:
                Ab[i,] = Ab[i,] / Ab[i, enter]
        print(Ab, '：旋转后Ab')
        print(('此时基本量:', BV))
    if (tar == 'max'):
        print("基变量下标", BV)
        print(Ab[:Ab.shape[0], Ab.shape[1] - 1:Ab.shape[1]], "基变量取值")
        print("基变量价值系数", temp)
        print(Ab[:, :Ab.shape[1] - 1], "A矩阵")
        print("最优目标函数值", -np.dot(temp, Ab[:Ab.shape[0], Ab.shape[1] - 1:Ab.shape[1]]))
    else:
        print("基变量下标", BV)
        print(Ab[:Ab.shape[0], Ab.shape[1] - 1:Ab.shape[1]], "基变量取值")
        print("基变量价值系数", temp)
        print(Ab[:, :Ab.shape[1] - 1], "A矩阵")
        print("最优目标函数值", np.dot(temp, Ab[:Ab.shape[0], Ab.shape[1] - 1:Ab.shape[1]]))




c = np.array([2.0, 3.0, 4.0, 0, 0])
A = np.array([[-1, -1.0, -1, 1, 0], [-2.0, 1, -3, 0, 1]])
b = np.array([[-3], [-4]])
BV = np.array([3, 4])
tar='min'
Z = danchunXB(c, A, b, BV, tar)