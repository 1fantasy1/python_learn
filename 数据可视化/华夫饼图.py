import pandas as pd
from pywaffle import Waffle
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用更优雅的雅黑字体
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')

# 创建数据
data = {'上座': 150, '空座': 50}
df = pd.DataFrame.from_dict(data, orient='index', columns=['数量'])

# 计算百分比
total = df['数量'].sum()
percentages = (df['数量'] / total * 100).round(1)

# 自定义颜色
colors = ['#4C72B0', '#DD8452']

# 创建华夫饼图
fig = plt.figure(
    FigureClass=Waffle,
    rows=10,
    values=df['数量'],
    colors=colors,
    labels=[f'{label} ({value}, {percentages[label]}%)'
            for label, value in zip(df.index, df['数量'])],
    legend={
        'loc': 'upper center',
        'bbox_to_anchor': (0.5, -0.05),
        'ncol': len(df),
        'framealpha': 0,
        'fontsize': 10
    },
    figsize=(8, 6),
    block_aspect_ratio=1  # 确保方块是正方形
)

# 添加标题和注释
title = plt.title('某电影上座率分析', y=1.05, fontsize=14, fontweight='bold')
title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
