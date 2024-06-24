# 导入必要的库
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei，以支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据集，包含三种水果（香蕉，橙子，其他水果）的特征数据
datasets = {
    'banana': {'long': 400, 'not_long': 100, 'sweet': 350, 'not_sweet': 150, 'yellow': 450, 'not_yellow': 50},
    'orange': {'long': 0, 'not_long': 300, 'sweet': 150, 'not_sweet': 150, 'yellow': 300, 'not_yellow': 0},
    'other_fruit': {'long': 100, 'not_long': 100, 'sweet': 150, 'not_sweet': 50, 'yellow': 50, 'not_yellow': 150}
}


# 计算每种水果的总数以及所有水果的总数
def count_total(data):
    """
    计算每种水果的总数和所有水果的总数
    :param data: 数据集
    :return: 各种水果的总数字典，所有水果的总数
    """
    count = {}
    total = 0
    for fruit, features in data.items():
        # 每种水果的总数
        fruit_total = features['sweet'] + features['not_sweet']
        count[fruit] = fruit_total
        total += fruit_total
    return count, total


# 计算每种水果的先验概率
def cal_base_rates(data):
    """
    计算每种水果的先验概率
    :param data: 数据集
    :return: 每种水果的先验概率字典
    """
    categories, total = count_total(data)
    base_rates = {}
    for label, count in categories.items():
        base_rates[label] = count / total
    return base_rates


# 计算每个特征值在已知水果下的概率（似然概率）
def likelihood_prob(data):
    """
    计算每个特征值在已知水果下的概率（似然概率）
    :param data: 数据集
    :return: 每个特征值在已知水果下的概率字典
    """
    count, _ = count_total(data)
    likelihood = {}
    for fruit, features in data.items():
        attr_prob = {}
        for attr, value in features.items():
            attr_prob[attr] = value / count[fruit]
        likelihood[fruit] = attr_prob
    return likelihood


# 计算每个特征值在所有水果中的概率（证据概率）
def evidence_prob(data):
    """
    计算每个特征值在所有水果中的概率（证据概率）
    :param data: 数据集
    :return: 每个特征值在所有水果中的概率字典
    """
    attrs = list(data[next(iter(data))].keys())
    count, total = count_total(data)
    evidence = {}
    for attr in attrs:
        attr_total = 0
        for fruit in data:
            attr_total += data[fruit][attr]
        evidence[attr] = attr_total / total
    return evidence


# 贝叶斯分类器类
class NaiveBayesClassifier:
    def __init__(self, data=datasets):
        """
        初始化贝叶斯分类器，计算先验概率、似然概率和证据概率
        :param data: 数据集
        """
        self._data = data
        self._labels = list(self._data.keys())
        self._priori_prob = cal_base_rates(self._data)  # 先验概率
        self._likelihood_prob = likelihood_prob(self._data)  # 似然概率
        self._evidence_prob = evidence_prob(self._data)  # 证据概率

    def get_label(self, length, sweetness, color):
        """
        根据给定的特征值预测水果类别
        :param length: 长度特征（1表示'long', 0表示'not_long'）
        :param sweetness: 甜度特征（1表示'sweet', 0表示'not_sweet'）
        :param color: 颜色特征（1表示'yellow', 0表示'not_yellow'）
        :return: 各种水果的后验概率字典
        """
        attrs = []
        if length == 1:
            attrs.append('long')
        else:
            attrs.append('not_long')
        if sweetness == 1:
            attrs.append('sweet')
        else:
            attrs.append('not_sweet')
        if color == 1:
            attrs.append('yellow')
        else:
            attrs.append('not_yellow')

        res = {}
        total_prob = 0  # 用于归一化后验概率的总概率

        for label in self._labels:
            prob = self._priori_prob[label]  # 获取某水果的先验概率
            for attr in attrs:
                prob *= self._likelihood_prob[label][attr] / self._evidence_prob[attr]
            res[label] = prob
            total_prob += prob

        # 归一化后验概率
        for label in res:
            res[label] /= total_prob

        return res


# 手动输入特征值并进行预测
def main():
    print("这是一个判断是否为香蕉或者橙子的程序，请输入水果的特征值：")
    length = int(input("长度（1表示'long', 其他表示'not_long'）："))
    sweetness = int(input("甜度（1表示'sweet', 其他表示'not_sweet'）："))
    color = int(input("颜色（1表示'yellow', 其他表示'not_yellow'）："))

    classifier = NaiveBayesClassifier()
    result = classifier.get_label(length, sweetness, color)
    print("预测结果：")
    print(f"香蕉的概率是：{result['banana']:.2f}")
    print(f"橙子的概率是：{result['orange']:.2f}")
    print(f"其他水果的概率是：{result['other_fruit']:.2f}")
    # 准备数据
    labels = ['香蕉', '橙子', '其他水果']
    sizes = [result['banana'], result['orange'], result['other_fruit']]
    colors = ['gold', 'orange', 'cyan']
    # 绘制饼状图
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')  # 使饼状图为正圆形
    plt.title('水果预测概率')
    plt.show()

# 运行主函数
if __name__ == "__main__":
    main()