import pandas as pd
import squarify
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
data = {
    '国家': ['美国', '土耳其', '英国', '德国', '加拿大', '西班牙', '意大利', '中国'],
    '人数': [529951, 52167, 79885, 126266, 23318, 163027, 152271, 83482]
}
df = pd.DataFrame(data)

# 创建不同的颜色 - 使用tab10色图，每个国家不同颜色
colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

# 设置画布
fig = plt.figure(figsize=(14, 10), facecolor='#f5f5f5')
ax = fig.add_subplot(111, facecolor='#f5f5f5')

# 绘制矩形树图
squarify.plot(
    sizes=df['人数'],
    label=[f"{country}\n{num:,}" for country, num in zip(df['国家'], df['人数'])],
    color=colors,
    alpha=0.85,
    text_kwargs={'fontsize': 12, 'color': 'black', 'weight': 'bold'},
    pad=True,
    bar_kwargs={'edgecolor': 'white', 'linewidth': 2},
    ax=ax
)

# 添加标题和注释
plt.title('2020年4月12日各国新冠肺炎确诊人数统计\n矩形树图',
          fontsize=18, pad=20, weight='bold', color='#333333')
plt.text(0.5, 0.02, '数据来源: 公开数据整理',
         ha='center', transform=fig.transFigure, fontsize=10, color='gray')

# 显示坐标轴（不再隐藏）
ax.set_xlim(0, 100)  # 设置x轴范围
ax.set_ylim(0, 100)  # 设置y轴范围
ax.set_xlabel('X轴', fontsize=12)
ax.set_ylabel('Y轴', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)  # 添加网格线

# 创建图例
patches = [plt.Rectangle((0,0),1,1, fc=color) for color in colors]
ax.legend(patches, df['国家'], title='国家',
          bbox_to_anchor=(1.05, 1), loc='upper left',
          borderaxespad=0., frameon=False)

# 调整布局
plt.subplots_adjust(right=0.8)  # 为图例留出空间

# 显示图形
plt.show()
