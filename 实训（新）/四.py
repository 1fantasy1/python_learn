import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from matplotlib import rcParams

# 设置字体为 SimHei（黑体），它支持中文
rcParams['font.sans-serif'] = ['SimHei']

# 解决负号 '-' 显示为方块的问题
rcParams['axes.unicode_minus'] = False

# 加载本地 CSV 数据
local_file_path = "google_trends_test_output.csv"
dataset = pd.read_csv(local_file_path)

# 删除 isPartial 列（如果存在）
if 'isPartial' in dataset.columns:
    del dataset['isPartial']

# 将日期列转换为日期时间格式并设置为索引
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.set_index('date', inplace=True)

# 生成统计描述并输出
describe = dataset.describe()
# print("数据统计描述：\n", describe)


def show_point_compare(df):
    """
    绘制虚线折线图，比较不同搜索词的趋势。

    参数:
    df (DataFrame): 包含搜索数据的 DataFrame，列为搜索词。
    """
    plt.figure(figsize=(15, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], '--', label=col)

    plt.title("Google 搜索趋势对比（虚线图）")
    plt.xlabel("日期")
    plt.ylabel("搜索频率")
    plt.legend()
    plt.grid("b--")
    plt.show()


def show_line_compare(df):
    """
    绘制实线折线图，比较不同搜索词的趋势。

    参数:
    df (DataFrame): 包含搜索数据的 DataFrame，列为搜索词。
    """
    plt.figure(figsize=(15, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title("Google 搜索趋势对比（实线图）")
    plt.xlabel("日期")
    plt.ylabel("搜索频率")
    plt.legend()
    plt.grid()
    plt.show()


def get_media_year(year):
    """
    筛选指定年份的数据并绘制折线图。

    参数:
    year (str): 指定的年份（如 '2018'）。
    """
    year = str(year)
    try:
        filtered_data = dataset.loc[dataset.index.year == int(year)]
        if filtered_data.empty:
            print(f"没有找到年份 {year} 的数据")
        else:
            print(f"年份 {year} 的数据统计描述：\n", filtered_data.describe())
            show_line_compare(filtered_data)
    except Exception as e:
        print(f"处理年份 {year} 数据时出错：{e}")


# 调用绘图函数
show_point_compare(dataset[['peanut butter', 'pizza', 'cookie']])
# get_media_year("2018")
# get_media_year("2019")
# get_media_year("2020")

def frequency_total(ano=None):
    """
    绘制搜索频率的饼状图。

    参数:
    ano (int 或 str，可选): 要筛选的年份。如果为 None，则统计所有年份的数据。
    """
    # 设置图形大小
    plt.figure(figsize=(15, 6))

    # 根据年份筛选数据，并生成标签
    if ano:
        ano = str(ano)  # 确保年份为字符串
        # 根据年份筛选数据
        filtered_data = dataset.loc[dataset.index.year == int(ano)]
        if filtered_data.empty:
            print(f"没有找到年份 {ano} 的数据")
            return  # 如果没有数据，退出函数
        title = f"各产品搜索占比 - {ano}"
        produtos_sum = filtered_data.sum()  # 计算筛选数据的总和
    else:
        title = "各产品搜索占比（所有年份）"
        produtos_sum = dataset.sum()  # 计算所有年份数据的总和

    # 动态生成产品名称列表
    list_product = produtos_sum.index

    # 绘制饼状图
    colors = plt.cm.Paired(range(len(list_product)))  # 使用预定义颜色方案
    plt.pie(
        produtos_sum,
        labels=list_product,
        autopct='%1.1f%%',  # 显示百分比
        startangle=90,      # 起始角度
        pctdistance=0.85,   # 百分比显示位置
        shadow=True,        # 添加阴影
        colors=colors       # 使用自定义颜色
    )

    # 添加中心白色圆圈，使饼图呈现环形
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # 图表细节设置
    plt.axis('equal')  # 确保饼图是圆形
    plt.title(title)   # 设置标题
    plt.tight_layout() # 调整布局避免重叠
    plt.show()

# frequency_total("2017")
# frequency_total("2018")
# frequency_total("2020")

import statsmodels.api as sm
import matplotlib.pyplot as plt


def decompose_and_plot(dataset, period=7):
    """
    对数据集中的每一列进行季节性分解并绘制图像。

    参数:
    dataset (DataFrame): 时间序列数据集，每列为一个序列。
    period (int): 时间序列的季节性周期。
    """
    for col in dataset.columns:
        try:
            # 对当前列进行季节性分解
            result = sm.tsa.seasonal_decompose(dataset[col].dropna(), period=period)
s
            # 绘制分解结果
            fig = result.plot()
            fig.set_figheight(9)
            fig.set_figwidth(14)
            fig.suptitle(f"Seasonal Decompose: {col}", fontsize=16)  # 添加标题

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以容纳标题
            plt.show()
        except Exception as e:
            print(f"列 {col} 分解失败：{e}")


# 调用函数，对数据集进行分解和绘图
# decompose_and_plot(dataset, period=7)
# decompose_and_plot(dataset, period=12)

