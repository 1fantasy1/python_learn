import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'retail_store_inventory.csv'
df = pd.read_csv(file_path)

# 显示数据框的前几行
# print(df.head())

# 将 'Date' 列转换为日期时间格式
df['Date'] = pd.to_datetime(df['Date'])

# 检查数据框中每列的缺失值数量
missing_values = df.isnull().sum()

# 输出每列的缺失值数量
# print(missing_values)

# 使用 seaborn 绘制库存水平的分布图
sns.histplot(df['Inventory Level'], bins=30, kde=True)  # bins=30 表示将数据分成30个区间，kde=True 添加核密度估计曲线
plt.title('库存水平的分布')  # 设置图表标题
plt.xlabel('库存水平')  # 设置 x 轴标签
plt.ylabel('频率')  # 设置 y 轴标签
plt.show()  # 显示图表

# 使用 seaborn 绘制产品类别分布图
sns.countplot(y='Category', data=df)  # y='Category' 表示在 y 轴上显示类别，data=df 指定数据来源为 df
plt.title('产品类别分布')  # 设置图表标题
plt.xlabel('数量')  # 设置 x 轴标签为“数量”
plt.ylabel('类别')  # 设置 y 轴标签为“类别”
plt.show()  # 显示图表

# 选择数值型列
numeric_df = df.select_dtypes(include=[np.number])  # 选择数据框中所有数值类型的列

# 绘制相关性热图
plt.figure(figsize=(12, 8))  # 设置图表的大小
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')  # 计算相关矩阵并绘制热图，annot=True 显示数值，fmt='.2f' 保留两位小数
plt.title('相关性热图')  # 设置图表标题
plt.show()  # 显示热图

from sklearn.model_selection import train_test_split  # 导入训练集和测试集的拆分方法
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
from sklearn.metrics import mean_squared_error  # 导入均方误差评价指标

# 定义特征和目标变量
features = numeric_df.drop(columns=['Units Sold'])  # 特征为数据框中的数值型列，除去 'Units Sold' 列
target = numeric_df['Units Sold']  # 目标变量为 'Units Sold' 列，即单位销售量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# 将数据按 80%/20% 的比例划分为训练集和测试集，random_state 设置为42确保结果可重复

# 初始化随机森林回归模型并训练
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 使用100颗树的随机森林回归模型
model.fit(X_train, y_train)  # 使用训练集数据进行模型训练

# 在测试集上进行预测
y_pred = model.predict(X_test)  # 使用训练好的模型对测试集进行预测，返回预测值

# 计算均方误差 (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)  # 计算真实值与预测值之间的均方误差

# 计算均方根误差 (Root Mean Squared Error)
rmse = np.sqrt(mse)  # 对均方误差开平方得到均方根误差，RMSE 更易于解释

print(rmse)  # 输出均方根误差
