import pandas as pd
import matplotlib.pyplot as plt

local_file_path = "google_trends_test_output.csv"

# 从本地 CSV 文件加载数据
dataset = pd.read_csv(local_file_path)

# 删除数据集中的 'isPartial' 列（如果存在）
if 'isPartial' in dataset.columns:
    del dataset['isPartial']

# 生成统计描述并输出
describe = dataset.describe()

# 输出统计描述到命令行
print(describe)

def show_point_compare(df):
    # 只选择需要的列，去除 'date' 和 'isPartial' 列
    df = df[['peanut butter', 'pizza', 'cookie']]
    plt.figure(figsize=(15, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], '--')

    plt.title("Search")
    plt.legend(df.columns)
    plt.xlabel("data")
    plt.ylabel("frequency")
    plt.grid("b--")
    plt.show()
show_point_compare(dataset)

def show_line_compare(df):
    plt.figure(figsize=(15, 6))
    for col in df.columns:
        plt.plot(df.index, df[col])

    plt.title("Search")
    plt.legend(df.columns)
    plt.xlabel("data")
    plt.ylabel("frequency")
    plt.grid()
    plt.show()


def get_media_year(ano):
    """
    提取指定年份的数据并绘制折线图。

    参数:
    ano (str): 要提取的年份（如 '2018'）。
    """
    # 确保 `ano` 是字符串格式的年份
    ano = str(ano)

    # 使用布尔索引筛选数据框中属于指定年份的行
    y_index = dataset[dataset.index.year == int(ano)]

    # 检查是否有数据
    if y_index.empty:
        print(f"没有找到年份 {ano} 的数据")
    else:
        # 调用绘图函数绘制筛选后的数据
        show_line_compare(y_index)
# 将日期列转换为日期时间格式并设置为索引
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.set_index('date', inplace=True)
get_media_year("2018")
get_media_year("2019")
get_media_year("2020")