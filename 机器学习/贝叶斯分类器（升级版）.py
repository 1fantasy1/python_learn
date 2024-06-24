import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei，以支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取 CSV 文件
file_path = 'data.csv'
data = pd.read_csv(file_path)

# 类别名称的映射
class_names = {
    'BARBUNYA': '巴尔布尼亚豆',
    'BOMBAY': '孟买豆',
    'CALI': '卡利豆',
    'DERMASON': '德尔马森豆',
    'HOROZ': '霍罗兹豆',
    'SEKER': '塞克尔豆',
    'SIRA': '西拉豆'
}

# 准备数据集
def prepare_datasets(data):
    classes = data['Class'].unique()
    datasets = {}
    for bean_type in classes:
        bean_data = data[data['Class'] == bean_type]
        datasets[bean_type] = {
            '面积': bean_data['Area'].tolist(),
            '周长': bean_data['Perimeter'].tolist(),
            '长轴长度': bean_data['MajorAxisLength'].tolist(),
            '短轴长度': bean_data['MinorAxisLength'].tolist(),
            '纵横比': bean_data['AspectRatio'].tolist(),
            '偏心率': bean_data['Eccentricity'].tolist(),
            '凸包面积': bean_data['ConvexArea'].tolist(),
            '等效直径': bean_data['EquivDiameter'].tolist(),
            '范围': bean_data['Extent'].tolist(),
            '密实度': bean_data['Solidity'].tolist(),
            '圆度': bean_data['Roundness'].tolist(),
            '紧致度': bean_data['Compactness'].tolist(),
            '形状因子1': bean_data['ShapeFactor1'].tolist(),
            '形状因子2': bean_data['ShapeFactor2'].tolist(),
            '形状因子3': bean_data['ShapeFactor3'].tolist(),
            '形状因子4': bean_data['ShapeFactor4'].tolist()
        }
    return datasets

datasets = prepare_datasets(data)

# 获取特征值的范围
def get_feature_ranges(data):
    ranges = {}
    for column in data.columns:
        if column != 'Class':
            ranges[column] = (data[column].min(), data[column].max())
    return ranges

feature_ranges = get_feature_ranges(data)

# 计算每种干豆类型的总数以及所有干豆的总数
def count_total(data):
    count = {}
    total = 0
    for bean_type, features in data.items():
        bean_total = len(features['面积'])
        count[bean_type] = bean_total
        total += bean_total
    return count, total

# 计算每种干豆类型的先验概率
def cal_base_rates(data):
    categories, total = count_total(data)
    base_rates = {}
    for label, count in categories.items():
        base_rates[label] = count / total
    return base_rates

# 计算每个特征值在已知干豆类型下的概率（似然概率）
def likelihood_prob(data):
    count, _ = count_total(data)
    likelihood = {}
    for bean_type, features in data.items():
        attr_prob = {}
        for attr, values in features.items():
            attr_prob[attr] = sum(values) / count[bean_type]
        likelihood[bean_type] = attr_prob
    return likelihood

# 计算每个特征值在所有干豆类型中的概率（证据概率）
def evidence_prob(data):
    attrs = list(data[next(iter(data))].keys())
    count, total = count_total(data)
    evidence = {}
    for attr in attrs:
        attr_total = 0
        for bean_type in data:
            attr_total += sum(data[bean_type][attr])
        evidence[attr] = attr_total / total
    return evidence

# 贝叶斯分类器类
class NaiveBayesClassifier:
    def __init__(self, data=datasets):
        self._data = data
        self._labels = list(self._data.keys())
        self._priori_prob = cal_base_rates(self._data)  # 先验概率
        self._likelihood_prob = likelihood_prob(self._data)  # 似然概率
        self._evidence_prob = evidence_prob(self._data)  # 证据概率

    def get_label(self, features):
        res = {}
        total_prob = 0  # 用于归一化后验概率的总概率

        for label in self._labels:
            prob = self._priori_prob[label]  # 获取某干豆类型的先验概率
            for attr, value in features.items():
                prob *= self._likelihood_prob[label][attr] / self._evidence_prob[attr]
            res[label] = prob
            total_prob += prob

        # 归一化后验概率
        for label in res:
            res[label] /= total_prob

        return res

# 手动输入特征值并进行预测
def main():
    print("这是一个干豆分类程序，请输入干豆的特征值：")
    features = {}
    feature_names = {
        'Area': '面积',
        'Perimeter': '周长',
        'MajorAxisLength': '长轴长度',
        'MinorAxisLength': '短轴长度',
        'AspectRatio': '纵横比',
        'Eccentricity': '偏心率',
        'ConvexArea': '凸包面积',
        'EquivDiameter': '等效直径',
        'Extent': '范围',
        'Solidity': '密实度',
        'Roundness': '圆度',
        'Compactness': '紧致度',
        'ShapeFactor1': '形状因子1',
        'ShapeFactor2': '形状因子2',
        'ShapeFactor3': '形状因子3',
        'ShapeFactor4': '形状因子4'
    }

    for feature, (min_val, max_val) in feature_ranges.items():
        features[feature_names[feature]] = float(input(f"{feature_names[feature]} (范围: {min_val:.5f} - {max_val:.5f}): "))

    classifier = NaiveBayesClassifier()
    result = classifier.get_label(features)
    print("预测结果：")
    for bean_type, probability in result.items():
        print(f"{class_names[bean_type]}的概率是：{probability:.2f}")

    # 准备数据
    labels = [class_names[label] for label in result.keys()]
    sizes = list(result.values())
    colors = ['gold', 'orange', 'cyan', 'green', 'blue', 'red', 'purple'][:len(labels)]

    # 绘制饼状图
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')  # 使饼状图为正圆形
    plt.title('干豆预测概率')
    plt.show()

# 运行主函数
if __name__ == "__main__":
    main()
