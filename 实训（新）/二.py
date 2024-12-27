# 导入所需的库
import time
import numpy as np
import pandas as pd
from pytrends.request import TrendReq

# 模拟产品名称
test_samples = {'sample': ["peanut butter", "pizza", "cookie"]}

# 初始化 pytrends 请求对象
pytrend = TrendReq()


# 定义获取 Google 趋势数据的函数
def get_google_trends_data(list_product, time_start, time_end, state=None, country='US'):
    # 如果指定了州，则构建包含国家和州的区域代码
    if state:
        sigla = '{}-{}'.format(country, state)
    else:
        # 如果未指定州，则只使用国家代码
        sigla = '{}'.format(country)

    # 组合时间范围字符串
    data_composer = '{} {}'.format(time_start, time_end)

    # 构建 Google 趋势数据的请求负载
    pytrend.build_payload(
        kw_list=list_product,  # 关键词列表
        geo=sigla,  # 地理位置（国家或国家-州）
        cat=0,  # 类别（0表示所有类别）
        timeframe=data_composer  # 时间范围
    )
    time.sleep(5)  # 每次请求后暂停5秒，避免触发限制
    # 返回趋势数据，按时间分布
    return pytrend.interest_over_time()


# 模拟的测试用例：关键词列表和时间范围
list_product = test_samples['sample']
time_start = '2014-01-01'
time_end = '2020-08-01'

# 获取 Google 趋势数据
dataset = get_google_trends_data(list_product=list_product, time_start=time_start, time_end=time_end)


# 测试导出数据为 CSV 文件的功能
def export_data_csv(dataset, save_output):
    dataset.to_csv(save_output, sep=",")  # 使用逗号作为分隔符


# 测试是否成功获取到数据并保存为 CSV
test_csv_path = "google_trends_test_output.csv"
export_data_csv(dataset, test_csv_path)

# 打印文件路径，确认文件是否生成
print(f"CSV 文件已保存至：{test_csv_path}")

# 删除数据集中的 'isPartial' 列
if 'isPartial' in dataset.columns:
    del dataset['isPartial']

# 测试对数据进行样式处理（显示背景渐变）
styled_dataset = dataset.style.background_gradient(cmap='Greens')

# 生成统计描述并应用渐变色样式
describe = dataset.describe()
styled_describe = describe.style.background_gradient(cmap='Greens')

# 输出统计描述，查看前几行数据
print(describe.head())

# 返回处理好的数据和样式，供后续使用
dataset, styled_dataset, describe, styled_describe
