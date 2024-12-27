# # 导入 Google 趋势数据的请求模块
from pytrends.request import TrendReq
#
# # 配置绘图的参数
# from pylab import rcParams
#
# # 用于交互式小部件的库
# from ipywidgets import interact, interactive, fixed, interact_manual
#
# # 忽略警告信息
# import warnings
#
# # 导入 itertools 模块，用于高效迭代
# import itertools
#
# # 导入数值计算库 NumPy
# import numpy as np
#
# # 导入 Matplotlib 库用于数据可视化
import matplotlib.pyplot as plt
#
# # 导入交互式小部件库
# import ipywidgets as widgets
#
# # 导入 Pandas 库用于数据处理
# import pandas as pd
#
# # 导入 Statsmodels 模块用于统计建模
# import statsmodels.api as sm
#
# # 导入 Matplotlib 的基础设置模块
# import matplotlib
#
# # 导入 Plotly 的离线绘图功能
# import plotly.offline as py
#
# # 导入 Plotly 的图形对象
# import plotly.graph_objs as go

import time

# 导入 Seaborn 库用于高级数据可视化
import seaborn as sns

# 更新 Matplotlib 全局字体大小
plt.rcParams.update({'font.size': 9})

# 设置 Seaborn 的绘图风格为深色网格
sns.set(style="darkgrid")

# 初始化 Google 趋势请求对象
pytrend = TrendReq()

# 示例数据字典，包含一些产品名称
test_samples = {'sample': ["peanut butter", "pizza", "cookie"]}

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


# 从示例数据中提取产品关键词列表
list_product = test_samples['sample']

# 获取指定关键词的 Google 趋势数据，时间范围从 2014-01-01 到 2020-08-01
# dataset = get_google_trends_data(list_product=list_product, time_start='2020-08-01', time_end='2020-08-05')
dataset = get_google_trends_data(list_product=list_product, time_start='2014-01-01', time_end='2020-08-01')

# 定义导出数据为 CSV 文件的函数
def export_data_csv(dataset, save_output):
    # 使用 Pandas 的 to_csv 方法将数据集保存为 CSV 文件
    dataset.to_csv(save_output, sep=",")  # 使用逗号作为分隔符

# 删除数据集中名为 'isPartial' 的列
del dataset['isPartial']

# 使用 Pandas 的样式功能，将数据框的背景颜色按值大小进行渐变处理
dataset.style.background_gradient(cmap='Greens')

export_data_csv(dataset, 'google_trends_data.csv')

# 使用 describe() 方法生成数据集的统计描述信息
describe = dataset.describe()

# 使用样式功能，按数值大小对统计描述信息应用渐变背景色
describe.style.background_gradient(cmap='Greens')
