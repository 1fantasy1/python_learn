import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import operator

# 设置字体为 SimHei，以支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
有人在约会网站寻找适合自己的约会对象，约会网站会推荐不同的人选，经过一番总结，发现曾交往过三种类型的人：

"'didntLike(不喜欢)', 'smallDoses(魅力一般的人)', 'largeDoses(极具魅力的人)'"

它将收集约会数据存放在文本文件 '求之不得表.txt' 中，每个样本数据占据一行，总共有 1000 行。样本主要包含以下 3 种特征：

每年获得的飞行常客里程数
玩视频游戏所耗时间百分比
每周消费的冰淇淋公升数

"""

def file2matrix(filename):
    """
    :param filename: 数据集的地址
    :return: 返回一个特征集和一个分类标签集
    """
    # 打开文件并读取所有行
    with open(filename, 'r', encoding='utf-8') as f:
        arrayLines = f.readlines()

    # 创建一个全零矩阵，行数与文件行数相同，列数为3
    returnMat = np.zeros((len(arrayLines), 3))
    # 创建一个空列表用于存储标签
    labelsVector = []
    # 初始化索引
    index = 0

    # 遍历每一行
    for line in arrayLines:
        # 去除换行符并用制表符分割
        features_label = line.replace("\n", "").split('\t')
        # 将前三个元素（特征）存入矩阵中
        returnMat[index, :] = features_label[:3]

        # 根据标签的值，将其转换为整数标签并存入列表中
        if features_label[-1] == 'largeDoses':
            labelsVector.append(int(3))
        elif features_label[-1] == 'smallDoses':
            labelsVector.append(int(2))
        else:
            labelsVector.append(int(1))

        # 索引加1
        index += 1

    return returnMat, labelsVector

def autoNorm(dataset):
    """
    :param dataset: 特征集
    :return: 返回经过标准归一化之后的特征集和每个特征的最大最小差值集合,和每个特征最小值的集合
    """
    # 计算每个特征的最小值
    minVals = dataset.min(0)

    # 计算每个特征的最大值
    maxVals = dataset.max(0)

    # 计算每个特征的范围（最大值 - 最小值）
    ranges = maxVals - minVals

    # 获取数据集中样本的数量
    m = dataset.shape[0]

    # 将每个特征的最小值扩展到与dataset相同的形状
    normDataSet = dataset - np.tile(minVals, (m, 1))

    # 将每个特征的范围扩展到与dataset相同的形状，并进行归一化
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    # 返回归一化后的数据集、每个特征的范围和每个特征的最小值(后面两个仅用于测试输出调试)
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, K):
    """
    :param inX: 目标特征集
    :param dataSet: 特征数据集
    :param labels: 对应特征数据集的标签数据集
    :param K: 最近邻K的k值
    :return: 返回目标特征集经过k-近邻算法所得出的分类结果
    """
    # 获取数据集样本数量
    dataSetSize = dataSet.shape[0]

    # 计算目标特征集与数据集中每个样本的差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # 计算差值的平方
    sqDiffMat = diffMat ** 2

    # 计算每个差值向量的平方和
    sqDistances = sqDiffMat.sum(axis=1)

    # 计算距离（欧几里得距离）
    distances = sqDistances ** 0.5

    # 按距离从小到大排序，返回排序后的索引值
    sortedDistIndices = distances.argsort()

    # 创建一个字典，用于存储K个最近邻的类别及其出现次数
    classCount = {}

    # 统计前K个最近邻的类别出现次数
    for i in range(K):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 按照类别出现次数降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    """
    classCount.items() 将 classCount 字典转换为包含键值对的元组列表，例如 [(key1, value1), (key2, value2), ...]。
    key=operator.itemgetter(1) 指定按照元组的第二个元素（即值，即类别出现次数）进行排序。
    reverse=True 表示降序排序，即出现次数最多的类别在最前面。
    """
    # 返回出现次数最多的类别
    return sortedClassCount[0][0]

def datingClassTest():
    """
    :return: 对算法模型进行测试,无返回值
    """
    # 设置测试集比例
    testRadio = 0.1

    # 调用函数读取数据
    datingDataMat, datingDataLabels = file2matrix("求之不得表.txt")

    # 创建图形
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制散点图
    scatter = ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
                         1.0 * np.array(datingDataLabels),
                         1.0 * np.array(datingDataLabels))
    plt.xlabel('玩视频游戏所消耗时间百分比')  # X轴标签
    plt.ylabel('每周消费的冰激淋公升数')  # Y轴标签
    plt.title('散点图示例')  # 图标题
    plt.show()

    # 对数据集进行归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 获取数据集样本的数量
    m = normMat.shape[0]

    # 计算测试集样本数量
    testNum = int(m * testRadio)

    # 初始化错误计数
    error = 0

    # 遍历测试集样本
    for i in range(testNum):
        # 使用KNN算法进行分类预测
        predict_result = classify0(normMat[i, :], normMat[testNum:m, :], datingDataLabels[testNum:m], 3)

        # 打印每条测试数据的预测结果和真实结果
        print(f"第{i}条数据的预测结果为: ", predict_result, ",其真实结果为: ", datingDataLabels[i])

        # 如果预测结果与真实结果不一致，错误计数加1
        if predict_result != datingDataLabels[i]:
            error += 1

    # 计算并打印模型预测的准确率
    print("模型预测的准确率为: ", (m - error) / m)

# datingClassTest()

def classifyPerson():

    resultset = ['didntLike(不喜欢)', 'smallDoses(魅力一般的人)', 'largeDoses(极具魅力的人)']

    # 加载数据集和标签，并进行归一化处理
    datingDataMat, datingDataLabels = file2matrix("求之不得表.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 初始化特征向量为长度为3的零向量
    features = np.zeros(3)

    # 输入特征值
    features[0] += float(input("请输入他(她)每年获得的飞行常客里程数: "))
    features[1] += float(input("请输入他(她)玩视频游戏所耗时间百分比: "))
    features[2] += float(input("请输入他(她)每周消费的冰淇淋公升数: "))

    # 对输入的特征向量进行归一化处理，并使用KNN算法进行分类预测
    normFeatures = (features - minVals) / ranges
    result = classify0(normFeatures, normMat, datingDataLabels, 3)

    # 打印预测结果
    print("该人可能属于 ", resultset[result - 1], " 类型")

if __name__ == '__main__':
    classifyPerson()