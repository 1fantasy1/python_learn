# 忽略警告信息（可在调试时开启以查看潜在问题）
import warnings
warnings.simplefilter('ignore')

# 导入必要的库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 导入机器学习相关库
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 数据编码与标准化
from sklearn.model_selection import train_test_split, cross_val_score, KFold  # 数据集划分与交叉验证

# 加载数据集
data = pd.read_csv('Airbnb_Data.csv')  # 从CSV文件中读取数据
df = data.copy()  # 创建数据副本，避免直接修改原始数据

# 输出数据的前5行，检查数据的基本结构
# print(df.head())

# 输出数据的形状（行数和列数）
# print(df.shape)

# 输出数据的列名，帮助识别特征
# print(df.columns)

# 数据描述性统计（如均值、标准差等），检查数值特征的分布
print(df.describe())

# 数据的基本信息（如数据类型和是否有缺失值）
# print(df.info())

# 数据类型概览
# print(df.dtypes)

# 检查每列的缺失值总数
# print(df.isnull().sum())

# 删除不需要的列，简化数据集
# 删除的列可能是与预测任务无关的（如文本描述、日期等），或是含有大量缺失值的列
new_df = df.drop(
    [
        'id', 'name', 'description', 'first_review', 'host_since', 
        'host_has_profile_pic', 'host_identity_verified', 'last_review', 
        'neighbourhood', 'thumbnail_url', 'zipcode', 'host_response_rate'
    ], 
    axis=1  # 指定按列删除
)

# 查看简化后的数据集的前5行
# print(new_df.head())

# 输出简化后的数据集的描述性统计信息
# print(new_df.describe())
print(new_df.columns)


# 输出简化后的数据集的形状（行数和列数）
# print(new_df.shape)

# 填充缺失值
# 对"bathrooms"列，用中位数的四舍五入值填充缺失值
new_df["bathrooms"] = df['bathrooms'].fillna(round(df["bathrooms"].median()))

# 对"review_scores_rating"列，用0填充缺失值（可能表示无评分）
new_df["review_scores_rating"] = df["review_scores_rating"].fillna(0)

# 对"bedrooms"列，用"bathrooms"列的中位数值填充缺失值（可能假设卫生间数量与卧室数量相关）
new_df["bedrooms"] = df['bedrooms'].fillna((df["bathrooms"].median()))

# 对"beds"列，用"bathrooms"列的中位数值填充缺失值（同样假设两者相关）
new_df["beds"] = df["beds"].fillna((df["bathrooms"].median()))

# 检查数据集中剩余的缺失值情况
# print(new_df.isnull().sum())
'''
# 定义绘制分类图的函数
def plot_catplot(h, v, he, a):
    """
    绘制分类变量图
    参数：
    h: 横轴变量
    v: 图类型（如 'bar', 'box' 等）
    he: 图的高度
    a: 图的宽高比（aspect ratio）
    """
    sns.set(font_scale=1.5)  # 设置字体大小
    sns.catplot(x=h, kind=v, data=df, height=he, aspect=a)  # 绘制分类图

# 定义绘制饼图的函数
def plot_piechart(h):
    """
    绘制饼图
    参数：
    h: 要统计的分类变量列名
    """
    sns.set(font_scale=1.5)  # 设置字体大小
    fig = plt.figure(figsize=(5, 5))  # 设置图的大小
    ax = fig.add_axes([0, 0, 1, 1])  # 添加图的坐标轴
    ax.axis('equal')  # 保证饼图为圆形
    langs = list(df[h].unique())  # 获取分类变量的唯一值列表
    students = list(df[h].value_counts())  # 获取每个分类的频率
    ax.pie(students, labels=langs, autopct='%1.2f%%')  # 绘制饼图并设置标签和百分比格式
    plt.show()  # 显示饼图

# 绘制价格分布直方图
plt.figure(figsize=(8, 6))  # 设置图的大小
sns.distplot(df["log_price"])  # 使用Seaborn绘制分布图
plt.title('Price distribution')  # 设置标题
# plt.show()  # 显示图形

# 绘制分类变量 "room_type" 的计数图
# plot_catplot("room_type", "count", 5, 2)
# 参数说明：
# - "room_type" 是横轴变量
# - "count" 表示绘制计数图
# - 5 是图的高度
# - 2 是宽高比

# 绘制 "room_type" 的饼图，展示不同房间类型的比例分布
# plot_piechart("room_type")
# 参数说明：
# - "room_type" 是需要绘制饼图的分类变量

# 绘制分类变量 "city" 的计数图，展示不同城市的房源数量
plot_catplot("city", "count", 5, 2)
# 参数说明与上类似，"city" 是横轴变量

# 绘制城市分布的饼图，展示每个城市房源的比例
fig = plt.figure(figsize=(5, 5))  # 设置图的大小为 5x5
ax = fig.add_axes([0, 0, 1, 1])  # 添加坐标轴
ax.axis('equal')  # 确保饼图为圆形

# 获取唯一城市名称列表
langs = list(df.city.unique())
# 获取每个城市的房源数量
students = list(df.city.value_counts())

# 绘制饼图
# ax.pie(students, labels=langs, autopct='%1.2f%%')
# 参数说明：
# - `students` 是每个分类的频率
# - `labels=langs` 为每块饼对应的标签
# - `autopct='%1.2f%%'` 显示百分比并保留两位小数

# 显示饼图
# plt.show()

# 获取 "neighbourhood" 列中出现频率最高的前15个社区
data = df.neighbourhood.value_counts()[:15]
# - `value_counts()` 计算每个社区的出现次数
# - `[:15]` 提取前15个社区

# 设置图形大小为 22x22
plt.figure(figsize=(22, 22))

# 将社区名称和对应的计数转换为列表
x = list(data.index)  # 社区名称
y = list(data.values)  # 每个社区的房源数量

# 翻转 x 和 y 列表，以便条形图按升序排列
x.reverse()
y.reverse()

# 设置图标题和轴标签
plt.title("Most popular Neighbourhood")  # 图标题
plt.ylabel("Neighbourhood Area")  # y轴标签
plt.xlabel("Number of guest who host in this area")  # x轴标签

# 绘制水平条形图
# plt.barh(x, y)  # `barh` 绘制水平条形图

# 显示图形
# plt.show()

# 使用 plot_catplot 绘制 "cancellation_policy" 的分类图
# plot_catplot("cancellation_policy", "count", 10, 2)
# 参数说明：
# - "cancellation_policy" 是横轴变量
# - "count" 表示绘制计数图
# - 10 是图的高度
# - 2 是宽高比

# 使用 plot_catplot 绘制 "cleaning_fee" 的分类图
# plot_catplot("cleaning_fee", "count", 6, 2)
# 参数说明与上类似，"cleaning_fee" 是横轴变量

# 定义绘制箱线图的函数
def plot_boxplot(h, v):
    """
    绘制箱线图
    参数：
    h: 横轴变量（分类变量）
    v: 纵轴变量（连续变量）
    """
    plt.figure(figsize=(10, 8))  # 设置图形大小
    sns.set(font_scale=1.5)  # 设置字体大小
    sns.boxplot(data=df, x=h, y=v, palette='GnBu_d')  # 绘制箱线图
    plt.title('Density and distribution of prices ', fontsize=15)  # 设置标题
    plt.xlabel(h)  # 设置横轴标签
    plt.ylabel(v)  # 设置纵轴标签
    plt.show()

# 分别绘制不同分类变量与价格的箱线图

# 绘制 "city" 与 "log_price" 的箱线图
# plot_boxplot("city", "log_price")
# 分析不同城市的价格分布情况

# 绘制 "room_type" 与 "log_price" 的箱线图
# plot_boxplot("room_type", "log_price")
# 分析不同房间类型的价格分布

# 绘制 "cancellation_policy" 与 "log_price" 的箱线图
# plot_boxplot("cancellation_policy", "log_price")
# 分析不同取消政策下的价格分布

# 绘制 "bed_type" 与 "log_price" 的箱线图
# plot_boxplot("bed_type", "log_price")
# 分析不同床类型的价格分布

# 使用 plot_catplot 绘制 "bed_type" 的分类计数图
# plot_catplot("bed_type", "count", 8, 2)
# 展示不同床类型的房源数量分布

# 初始化存储分类变量和数值变量的列表
categorical_col = []  # 用于存储分类变量的列名
numerical_col = []  # 用于存储数值变量的列名

# 遍历数据集的所有列，根据数据类型将列分类
for column in new_df.columns:
    if new_df[column].dtypes != "float64" and new_df[column].dtypes != "int64":
        categorical_col.append(column)  # 如果不是数值类型，添加到分类变量列表
    else:
        numerical_col.append(column)  # 如果是数值类型，添加到数值变量列表

# 输出分类变量和数值变量的列名
# print(numerical_col)  # 打印数值变量列名
# print(categorical_col)  # 打印分类变量列名

# 使用 LabelEncoder 对分类变量进行编码
le = LabelEncoder()
for col in categorical_col:
    new_df[col] = le.fit_transform(new_df[col])  # 将分类变量转换为数值类型

# 设置 Pandas 显示选项，显示所有列
pd.set_option("display.max_columns", None)

# 输出经过编码处理的数据集
# print(new_df)

# 划分特征和目标变量
x = new_df.drop('log_price', axis=1)  # 特征集（去掉目标列 'log_price'）
y = new_df['log_price']  # 目标变量（价格列）

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# - `test_size=0.2` 表示 20% 的数据作为测试集
# - `random_state=42` 保证每次运行代码时划分结果一致

# 对特征进行标准化处理
sc = StandardScaler()  # 初始化标准化工具
x_train = sc.fit_transform(x_train)  # 对训练集进行标准化（均值为0，方差为1）
x_test = sc.transform(x_test)  # 使用同样的参数对测试集进行标准化
'''
