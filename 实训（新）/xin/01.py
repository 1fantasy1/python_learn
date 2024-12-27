import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 设置 matplotlib 显示中文字体，防止出现乱码
rcParams['font.sans-serif'] = ['SimHei']

# 读取 YouTube 数据集
# 数据集文件名为 "YOUTUBE CHANNELS DATASET.csv"
df = pd.read_csv("YOUTUBE CHANNELS DATASET.csv")

# 查看数据集前五行，便于快速了解数据结构和内容
# print(df.head(5))

# 统计描述数据集的基本信息（如均值、中位数等），以便分析特征分布
# print(df.describe().T)

# 显示数据集的详细信息，包括列名、数据类型、非空值数量等，便于了解数据质量
# print(df.info())

# 定义数据预处理函数
# 将带有单位（如"M"、"B"）或逗号分隔的数字转换为整数类型
def convert_to_int(value):
    if isinstance(value, str):  # 如果值是字符串类型
        # 替换字符串中的逗号，"M" 表示百万（1e6），"B" 表示十亿（1e9）
        value = value.replace(',', '').replace('M', 'e6').replace('B', 'e9')
        return int(float(value))  # 将字符串转换为浮点数后再转为整数
    return value  # 如果值不是字符串，则直接返回原值

# 对 "Subscribers" 列进行预处理，将订阅者数转换为整数类型
df['Subscribers'] = df['Subscribers'].apply(convert_to_int)

# 对 "Uploads" 列进行预处理，将上传视频数转换为整数类型
df['Uploads'] = df['Uploads'].apply(convert_to_int)

# 对 "Views" 列进行预处理，将观看次数转换为整数类型
df['Views'] = df['Views'].apply(convert_to_int)

# 可视化前10个订阅者最多的频道
plt.figure(figsize=(10, 6))
top_channels = df.nlargest(10, 'Subscribers')
sns.barplot(data=top_channels, x='Subscribers', y='Username', hue='Username', palette='coolwarm', dodge=False)
plt.title("前10个订阅者最多的频道", fontsize=16)
plt.xlabel("订阅者（以百万计）", fontsize=12)
plt.ylabel("频道名称", fontsize=12)
plt.tight_layout()
# plt.show()

# 绘制饼图显示按国家分布的频道
plt.figure(figsize=(8, 8))
country_counts = df['Country'].dropna().value_counts()

# 设置爆炸效果，仅对最大值进行爆炸
explode = [0.1 if count == max(country_counts) else 0 for count in country_counts]

# 绘制饼图
country_counts.plot.pie(autopct='%1.1f%%', startangle=140, cmap='Set3', explode=explode)
plt.title("按国家分布的频道", fontsize=16)
plt.ylabel("")  # 隐藏y轴标签
plt.tight_layout()
plt.show()

# 绘制散点图显示各国上传视频数量与总观看量的关系
# df['Views'] = df['Views'] / 1e8  # 将观看量从原始单位转换为以“亿”为单位
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Uploads', y='Views', hue='Country', style='Country', palette='tab10', s=30)
plt.title("各国上传视频数量与总观看量的关系", fontsize=16)
plt.xlabel("上传视频数量", fontsize=12)
plt.ylabel("总观看量（以亿计）", fontsize=12)
plt.legend(title="国家", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 检查缺失值数量
# print(df.isnull().sum())

# 删除包含缺失值的行
df = df.dropna()

# 对 "Country" 列进行独热编码处理，包括缺失值的处理
df = pd.get_dummies(df, columns=['Country'], dummy_na=True)

# 构建特征和目标变量
X = df[['Uploads', 'Views'] + [col for col in df.columns if 'Country_' in col]]
y = df['Subscribers']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林回归模型并进行训练
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 测试集预测和误差计算
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
# print(f"平均绝对误差（Mean Absolute Error）: {mae}")

# 对目标变量应用对数变换
df['LogSubscribers'] = np.log1p(df['Subscribers'])

# 使用对数变换后的目标变量重新训练模型
y = df['LogSubscribers']
model.fit(X_train, np.log1p(y_train))

# 预测时对变换进行逆变换
y_pred = np.expm1(model.predict(X_test))

# 特征工程：新增特征
df['Subscribers_per_Upload'] = df['Subscribers'] / (df['Uploads'] + 1)
df['Views_per_Subscriber'] = df['Views'] / (df['Subscribers'] + 1)

# 打印数据前五行
# print("数据前五行：")
# print(df.head(5))

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 用测试集进行预测
# y_pred = model.predict(X_test)

# 计算平均绝对误差
mae = mean_absolute_error(y_test, y_pred)
# print(f"平均绝对误差（Mean Absolute Error）：{mae}")

# 选择特征和目标变量
features = df[['Ranking', 'Uploads', 'Views'] + [col for col in df.columns if 'Country' in col]]
target = df['Subscribers']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 特征标准化（对某些算法非常重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test_scaled)

# 创建实际值与预测值的 DataFrame
predictions = pd.DataFrame({'实际值': y_test, '预测值': y_pred})

# 计算评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出评估指标
print(f"平均绝对误差（Mean Absolute Error）：{mae}")
print(f"均方误差（Mean Squared Error）：{mse}")
print(f"R-squared（决定系数）：{r2}")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='预测值')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='理想线')  # 理想线（实际值=预测值）
plt.xlabel('实际订阅者数量', fontsize=12)
plt.ylabel('预测订阅者数量', fontsize=12)
plt.title('实际值与预测值对比', fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()
