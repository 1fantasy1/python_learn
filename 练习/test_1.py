# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from pyparsing import results
#
# # 设置字体
# rcParams['font.sans-serif'] = ['Microsoft YaHei']
#
# class BeanClassifier:
#     def __init__(self, file_path):
#         self.data = pd.read_csv(file_path)
#         self.prepare_datasets()
#         self.calculate_probabilities()
#
#     def prepare_datasets(self):
#         self.classes = self.data['Class'].unique()
#         self.datasets = {cls: self.data[self.data['Class'] == cls].iloc[:1:] for cls in self.classes}
#
#     def calculate_probabilities(self):
#         self.base_rates = self.calculate_base_rates() self.likelihood_probs = self.calculate_likelihood_probs()
#         self.evidence_probs = self.calculate_evidence_probs()
#
#     def calculate_base_rates(self):
#         counts = {cls: len(data) for cls, data in self.datasets.items()}
#         total = sum(counts.values()) return {cls: count / total for cls, count in counts.items()}
#
#     def calculate_likelihood_probs(self):
#         likelihoods = {}
#         for cls, data in self.datasets.items():
#             likelihoods[cls] = data.mean()
#         return likelihoods
#
#     def calculate_evidence_probs(self):
#         evidences = {}
#         for attr in self.datasets[next(iter(self.datasets))].columns:
#             evidences[attr] = sum(data[attr].sum() for data in self.datasets.values()) / sum(self.data[attr])
#         return evidences
#
#     def predict(self, features): results = {}
#         for cls, likelihoods in self.likelihood_probs.items():
#             prob = self.base_rates[cls]
#             for attr, value in features.items():
#                 prob *= likelihoods[attr] / self.evidence_probs[attr]
#             results[cls] = prob
#         total_prob = sum(results.values())
#         for cls in results:
#             results[cls] /= total_prob
#         return results
#
# def main():
#     classifier = BeanClassifier('data.csv')
#     features = {
#         'Area': float(input('面积: ')),
#         'Perimeter': float(input('周长: ')),
#         # 添加其他特征的输入
#     }
#     result = classifier.predict(features)
#     print("预测结果：")
#     for cls, probability in result.items():
#         print(f"{cls}的概率是：{probability:.2f}")
#
#     # 可视化部分省略，可根据需要添加
#
# if __name__ == "__main__":
#     main()
def compress_rle(data):
    compressed_data = []
    i = 0
    while i < len(data):
        count = 1
        while i < len(data) - 1 and data[i] == data[i + 1]:
            count += 1
            i += 1
        compressed_data.append((data[i], count))
        i += 1
    return compressed_data

# 示例数据
data = [10, 12, 12, 15, 15, 15, 255, 1, 2, 3, 3, 3]
compressed_data = compress_rle(data)
print("原始数据：", data)
print("压缩后的数据：", compressed_data)
def decompress_rle(compressed_data):
    decompressed_data = []
    for item in compressed_data:
        value, count = item
        decompressed_data.extend([value] * count)
    return decompressed_data

# 示例压缩数据
compressed_data = [(10, 1), (12, 2), (15, 3), (255, 1), (1, 1), (2, 1), (3, 3)]
decompressed_data = decompress_rle(compressed_data)
print("解压缩后的数据：", decompressed_data)
