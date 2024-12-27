# 导入需要的库
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from matplotlib.ticker import ScalarFormatter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder  # 导入标签编码器

# 设置 matplotlib 显示中文字体，防止出现乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 替换为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块

# 数据读取函数，读取CSV文件
df = pd.read_csv("001.csv", encoding="latin-1")

# 查看数据集前五行，便于快速了解数据结构和内容
# print(df.head(5))

# 统计描述数据集的基本信息（如均值、中位数等），以便分析特征分布
# print(df.describe().T)

# 显示数据集的详细信息，包括列名、数据类型、非空值数量等，便于了解数据质量
# print(df.info())

# 查看数据集中每列的空值数量
# print(df.isnull().sum())

# 假设`df`是你的DataFrame
df = df.copy()  # 对原始数据进行复制，以便修改而不影响原始数据

# 查看数据框中的列名
# print(df.columns)

# 删除不必要的列，这些列对模型分析没有帮助
df = df.drop(['rank', 'Abbreviation', 'country_rank', 'created_month',
             'created_date', 'Gross tertiary education enrollment (%)', 
             'Unemployment rate', 'Urban_population'], axis=1)

# 确认是否成功删除了所选的列
# print(df.columns)

# 使用 duplicated() 标记重复数据
duplicates = df.duplicated()

# 过滤出重复的行，检查是否存在重复数据
duplicated_rows = df[duplicates]

# 打印重复行，若没有重复行则为空
# print(duplicated_rows)

# 定义一个变量来检测空值（null）
nulls_count = df.isnull().sum()

# 过滤出含有空值的列，并显示空值数量
# print(nulls_count[nulls_count > 0])

# 选取数据类型为对象（即分类数据）的列
cat_colms = df.select_dtypes(include =['object']).columns

# 使用 fillna() 填充分类列中的缺失值，用'Unknown'替换
df[cat_colms]= df[cat_colms].fillna("Unknown")

# 选取数据类型为数值（整型和浮动型）的列
num_colms = df.select_dtypes(include = ['int64', 'float']).columns

# 使用 fillna() 填充数值列中的缺失值，用'0'替换
df[num_colms] = df[num_colms].fillna(0)

# 确认数据集中已没有空值
# print(df.isnull().sum())

# 打印数据框的信息，包括列名、数据类型和内存使用情况
# print(df.info())

''' 有些列不需要是浮点型，因此将它们转换为整型（int64）以节省内存空间和提高效率 '''
df = df.astype({
    'video views': 'int64',  # 视频观看次数
    'channel_type_rank': 'int64',  # 频道类型排名
    'video_views_rank': 'int64',  # 视频观看次数排名
    'video_views_for_the_last_30_days': 'int64',  # 最近30天的视频观看次数
    'subscribers_for_last_30_days': 'int64',  # 最近30天的订阅者数
    'created_year': 'int64',  # 创建年份
    'Population': 'int64',  # 人口数量
    'lowest_monthly_earnings': 'int64',  # 每月最低收入
    'highest_monthly_earnings': 'int64',  # 每月最高收入
    'lowest_yearly_earnings': 'int64',  # 每年最低收入
    'highest_yearly_earnings': 'int64'  # 每年最高收入
})

# 再次打印数据框信息，确认数据类型已更新
# print(df.info())

# 检查是否存在异常值，按 'video views' 分组并计数
# print(df.groupby('video views').size().head(5))

# 找出 'video views' 列中值为0的索引
zero_views_index = (df[df['video views'] == 0]).index
# print(zero_views_index)

# 使用这些索引删除值为0的行
df = df.drop(axis=0, index=zero_views_index)

# 验证删除后的结果，按 'video views' 分组并计数
# print(df.groupby('video views').size().head(5))

# 查看数据框前5行，以检查是否存在缺失的索引，例如索引1缺失
# print(df.head(5))

# 确保数据集按订阅者数量（subscribers）降序排列
df.sort_values(by='subscribers', ascending=False)

# 重置索引，drop=True 表示不会将旧索引添加为新的列
df = df.reset_index(drop=True)

# 打印数据框前5行，确认索引已重新排列
# print(df.head())

# 将清理后的数据框保存为 CSV 文件
# df.to_csv('002.csv', index=False, encoding='utf-8')
# print("数据已成功保存为 002.csv")

# 如果无法使用 seaborn 样式，改用 ggplot 样式
try:
    plt.style.use("seaborn")
except OSError:
    plt.style.use("ggplot")  # 使用备用样式

# 设置绘图尺寸
plt.rcParams['figure.figsize'] = (16, 8)

# 计算相关性矩阵
numeric_df = df.select_dtypes(include=[float, int])
correlation_matrix = numeric_df.corr()

# 绘制热力图
title = "Correlation Heatmap"
plt.title(title, fontsize=18, weight='bold')
sns.heatmap(correlation_matrix, cmap="BuPu", annot=True)
# plt.show()

# 统计每年创建频道数
channels_in_year = df['created_year'].value_counts()
channels_in_year = pd.DataFrame(channels_in_year).reset_index()
channels_in_year.columns = ['Year', 'Created Channels']

# 按年份排序
channels_in_year = channels_in_year.sort_values(by='Year', ascending=True)

# 删除异常年份数据（Year=0 和 Year=1970）
channels_in_year = channels_in_year[(channels_in_year['Year'] > 1970) & (channels_in_year['Year'] <= 2022)]

# 重置索引
channels_in_year = channels_in_year.reset_index(drop=True)

# 检查处理结果
# print(channels_in_year)

# 准备绘图数据
x = channels_in_year['Year']
y = channels_in_year['Created Channels']

# 设置画布大小
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制柱状图
ax.bar(x, y, color='#32A645', label='Number of Channels (Bar)')

# 绘制折线图
ax.plot(x, y, color='#004DFF', marker='o', label='Number of Channels (Line)')

# 添加坐标轴标签和标题
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Channels", fontsize=12)
ax.set_title("Number of Channels Created in Each Year", fontsize=14, weight='bold')

# 设置 x 和 y 的范围
ax.set_xlim(x.min() - 1, x.max() + 1)
ax.set_ylim(0, y.max() + 10)

# 添加图例
ax.legend()

# 显示图表
# plt.show()

# 统计频道类型分布
channel_types = df['channel_type'].value_counts()

# 动态筛选出占比小于 0.5% 的类型为 "Others"
threshold = 0.005
total_count = channel_types.sum()
filtered_channel_types = channel_types[channel_types / total_count >= threshold]
others_count = channel_types[channel_types / total_count < threshold].sum()
filtered_channel_types['Others'] = others_count

# 配置配色方案
colors = sns.color_palette('hls', len(filtered_channel_types))

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(
    filtered_channel_types,
    colors=colors,
    labels=filtered_channel_types.index,
    autopct=lambda p: f'{p:.1f}% ({int(p * total_count / 100)})'
)

# 设置标题
plt.title('Distribution of YouTube Channels by Type', weight='bold', fontsize=14)

# 显示图表
# plt.show()

# 获取国家频道分布，提取前 15 个
channel_orig = df['Country'].value_counts().head(15)
# print(channel_orig)

# 配置画布大小
plt.figure(figsize=(10, 8))

# 绘制饼图
plt.pie(
    channel_orig,
    labels=channel_orig.index,
    autopct=lambda p: f'{p:.1f}% ({int(p * channel_orig.sum() / 100)})',
    colors=sns.color_palette('hls', len(channel_orig))
)

# 设置标题
plt.title('Distribution of YouTube Channels by Country (Top 15)', weight='bold', fontsize=14)

# 显示图表
# plt.show()

# 选择前10条数据并排序
colms = ['Youtuber', 'subscribers']
bar_colms = df.loc[0:9, colms].sort_values('subscribers', ascending=True)

# 转换订阅人数为百万
bar_colms['subscribers (MM)'] = (bar_colms['subscribers'] / 1000000).astype(int)

# 打印数据查看
# print(bar_colms)

# 绘制图形
x = bar_colms['Youtuber']
y = bar_colms['subscribers (MM)']

# 图形大小
fig = plt.figure(figsize=(10, 6))

# 创建水平条形图
bars = plt.barh(x, y, color=sns.color_palette('hls', len(bar_colms)))

# 给每个条形添加数值标签
for i in range(len(bars)):
    plt.text(
        bars[i].get_width() + 0.1,  # 标签的横坐标位置
        bars[i].get_y() + bars[i].get_height() / 2,  # 标签的纵坐标位置
        f'{y.iloc[i]}M',  # 显示的文本
        va='center',  # 垂直居中
        fontsize=12,
        weight='bold'
    )

# 设置标签和标题
plt.xlabel("No. of Subscribers in Million", weight='bold', fontsize=12)
plt.ylabel("Youtuber", weight='bold', fontsize=12)
plt.title("Top 10 YouTube Channels by Subscribers", weight='bold', fontsize=14)

# 显示图表
# plt.show()

# 选择前10条数据并排序
colms = ['Youtuber', 'video views']
bar_colms = df.loc[0:9, colms].sort_values('video views', ascending=True)

# 转换视频观看次数为十亿
bar_colms['video views (bil)'] = (bar_colms['video views'] / 1000000000).astype(int)

# 打印数据查看
# print(bar_colms)

# 绘制图形
x = bar_colms['Youtuber']
y = bar_colms['video views (bil)']

# 图形大小
fig = plt.figure(figsize=(10, 6))

# 创建水平条形图
bars = plt.barh(x, y, color=sns.color_palette('hls', len(bar_colms)))

# 给每个条形添加数值标签
for i in range(len(bars)):
    plt.text(
        bars[i].get_width() + 0.05,  # 标签的横坐标位置
        bars[i].get_y() + bars[i].get_height() / 2,  # 标签的纵坐标位置
        f'{y.iloc[i]}B',  # 显示的文本
        va='center',  # 垂直居中
        fontsize=12,
        weight='bold'
    )

# 设置标签和标题
plt.xlabel("Total Video Views in Billion", weight='bold', fontsize=12)
plt.ylabel("Youtuber", weight='bold', fontsize=12)
plt.title("Top 10 YouTube Channels by Total Video Views", weight='bold', fontsize=14)

# 显示图表
# plt.show()

# 选择相关列
colms = ['subscribers', 'video views']

# 选择数据
scatter_colms = df.loc[0:, colms]

# 转换订阅数和视频观看次数为可读性更强的单位
scatter_colms['subscribers (MM)'] = (scatter_colms['subscribers'] / 1000000).astype(int)
scatter_colms['video views (bil)'] = (scatter_colms['video views'] / 1000000000).astype(int)

# print(scatter_colms)

# 获取x, y轴数据
x = scatter_colms['subscribers (MM)']
y = scatter_colms['video views (bil)']

# 设置点的大小
size = scatter_colms['subscribers (MM)'] * 0.5  # 增加放大倍数

# 绘制散点图
plt.figure(figsize=(10, 6))

# 通过散点图展示数据
plt.scatter(x, y, s=size, c=y, cmap='plasma', alpha=0.8, edgecolors="w", linewidth=0.5)

# 标题和标签
plt.title('Relationship Between Number of Subscribers & Total Video Views', fontsize=14, weight='bold')
plt.xlabel('Number of Subscribers (in millions)', fontsize=12, weight='bold')
plt.ylabel('Total Video Views (in billions)', fontsize=12, weight='bold')

# 通过颜色条展示视频观看次数的大小
plt.colorbar(label='Total Video Views (in billions)')

# 显示图表
# plt.show()

# 选择相关列
colms = ['video views', 'highest_yearly_earnings']

# 提取相关数据
scatter_colms = df.loc[0:, colms]

# 转换为更适合的单位
scatter_colms['video views (bill)'] = (scatter_colms['video views'] / 1000000000).astype(int)
scatter_colms['highest_yearly_earnings (MM)'] = (scatter_colms['highest_yearly_earnings'] / 1000000).astype(int)

# print(scatter_colms)

# 获取 x, y 数据
x = scatter_colms['video views (bill)']
y = scatter_colms['highest_yearly_earnings (MM)']

# 设置点的大小
size = scatter_colms['highest_yearly_earnings (MM)'] * 0.6  # 增加放大倍数

# 绘制散点图，使用颜色映射来表示视频观看次数
plt.figure(figsize=(10, 6))

# 使用更深的绿色调色板
plt.scatter(x, y, s=size, c=y, cmap='viridis', alpha=0.9, edgecolors="w", linewidth=0.5)

# 添加标题和标签
plt.title('Relationship Between Total Video Views & Highest Yearly Earning', fontsize=14, weight='bold')
plt.xlabel('Total Video Views (Billions)', fontsize=12, weight='bold')
plt.ylabel('Highest Yearly Earning (Millions)', fontsize=12, weight='bold')

# 显示颜色条，表示“最高年度收入”
plt.colorbar(label='Highest Yearly Earnings (Millions)')

# 显示图表
# plt.show()

# 选择相关列
colms = ['uploads', 'video views']

# 提取相关数据
scatter_colms = df.loc[0:, colms]

# 转换为更适合的单位
scatter_colms['video views (bil)'] = (scatter_colms['video views'] / 1000000000).astype(int)

# print(scatter_colms)

# 获取 x, y 数据
x = scatter_colms['video views (bil)']
y = scatter_colms['uploads']

# 设置点的大小，基于视频观看次数的大小
size = scatter_colms['video views (bil)'] * 0.6  # 增加放大倍数，调整点的大小

# 绘制散点图，使用颜色映射来表示视频观看次数
plt.figure(figsize=(10, 6))

# 使用颜色映射来根据视频观看次数着色，alpha=0.6 增加透明度
plt.scatter(x, y, s=size, c=x, cmap='cool_r', alpha=0.9, edgecolors="w", linewidth=0.5)

# 添加标题和标签
plt.title('Relationship Between Total Video Views & Total Uploads', fontsize=14, weight='bold')
plt.xlabel('Total Video Views (Billions)', fontsize=12, weight='bold')
plt.ylabel('Total Uploads', fontsize=12, weight='bold')

# 显示颜色条，表示视频观看次数的大小
plt.colorbar(label='Total Video Views (Billions)')

# 显示图表
plt.show()

