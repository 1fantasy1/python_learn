import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置 matplotlib 显示中文字体，防止出现乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置项
CONFIG = {
    'input_file': "001.csv",
    'output_file': "002.csv",
    'encoding': "latin-1",
    'figure_size': (16, 8),
    'font_family': 'Microsoft YaHei',
    'style': 'seaborn',
    'dpi': 300,
}


def setup_plotting_style():
    """设置绘图风格"""
    try:
        plt.style.use(CONFIG['style'])
    except OSError:
        plt.style.use('ggplot')


def load_and_preprocess_data(file_path, encoding):
    """加载并预处理数据"""
    df = pd.read_csv(file_path, encoding=encoding)

    # 删除不必要的列
    columns_to_drop = [
        'rank', 'Abbreviation', 'country_rank', 'created_month',
        'created_date', 'Gross tertiary education enrollment (%)',
        'Unemployment rate', 'Urban_population'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 处理缺失值
    fill_missing_values(df)

    # 转换数据类型
    convert_data_types(df)

    # 删除视频观看次数为0的行
    df = df[df['video views'] > 0]

    # 按订阅者数量降序排列并重置索引
    df.sort_values(by='subscribers', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fill_missing_values(df):
    """处理缺失值"""
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['int64', 'float']).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df[num_cols] = df[num_cols].fillna(0)


def convert_data_types(df):
    """转换数据类型"""
    type_conversions = {
        'video views': 'int64',
        'channel_type_rank': 'int64',
        'video_views_rank': 'int64',
        'video_views_for_the_last_30_days': 'int64',
        'subscribers_for_last_30_days': 'int64',
        'created_year': 'int64',
        'Population': 'int64',
        'lowest_monthly_earnings': 'int64',
        'highest_monthly_earnings': 'int64',
        'lowest_yearly_earnings': 'int64',
        'highest_yearly_earnings': 'int64'
    }
    df.astype(type_conversions, copy=False)


def save_cleaned_data(df, file_path, encoding):
    """保存清理后的数据"""
    df.to_csv(file_path, index=False, encoding=encoding)
    print(f"Data has been successfully saved to {file_path}")


def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
    numeric_df = df.select_dtypes(include=[float, int])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=CONFIG['figure_size'])
    sns.heatmap(correlation_matrix, cmap="BuPu", annot=True)
    plt.title("Correlation Heatmap", fontsize=18, weight='bold')
    plt.show()


def plot_channels_created_per_year(df):
    """统计每年创建频道数并绘图"""
    channels_in_year = df['created_year'].value_counts().reset_index()
    channels_in_year.columns = ['Year', 'Created Channels']
    channels_in_year = channels_in_year[(channels_in_year['Year'] > 1970) & (channels_in_year['Year'] <= 2022)]
    channels_in_year.sort_values(by='Year', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(channels_in_year['Year'], channels_in_year['Created Channels'], color='#32A645',
           label='Number of Channels (Bar)')
    ax.plot(channels_in_year['Year'], channels_in_year['Created Channels'], color='#004DFF', marker='o',
            label='Number of Channels (Line)')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Channels", fontsize=12)
    ax.set_title("Number of Channels Created in Each Year", fontsize=14, weight='bold')
    ax.legend()
    plt.show()


def plot_channel_type_distribution(df):
    """绘制频道类型分布饼图"""
    channel_types = df['channel_type'].value_counts()
    total_count = channel_types.sum()
    filtered_channel_types = channel_types[channel_types / total_count >= 0.005]
    others_count = channel_types[channel_types / total_count < 0.005].sum()
    filtered_channel_types['Others'] = others_count

    plt.figure(figsize=(8, 8))
    plt.pie(
        filtered_channel_types,
        labels=filtered_channel_types.index,
        autopct=lambda p: f'{p:.1f}% ({int(p * total_count / 100)})',
        colors=sns.color_palette('hls', len(filtered_channel_types))
    )
    plt.title('Distribution of YouTube Channels by Type', weight='bold', fontsize=14)
    plt.show()


def plot_country_distribution(df):
    """绘制国家频道分布饼图"""
    top_countries = df['Country'].value_counts().head(15)
    plt.figure(figsize=(10, 8))
    plt.pie(
        top_countries,
        labels=top_countries.index,
        autopct=lambda p: f'{p:.1f}% ({int(p * top_countries.sum() / 100)})',
        colors=sns.color_palette('hls', len(top_countries))
    )
    plt.title('Distribution of YouTube Channels by Country (Top 15)', weight='bold', fontsize=14)
    plt.show()


def plot_top_youtube_channels_by_subscribers(df):
    """绘制前10个YouTube频道的订阅者条形图"""
    colms = ['Youtuber', 'subscribers']
    bar_colms = df.loc[0:9, colms].sort_values('subscribers', ascending=True)
    bar_colms['subscribers (MM)'] = (bar_colms['subscribers'] / 1000000).astype(int)

    fig = plt.figure(figsize=(10, 6))
    bars = plt.barh(bar_colms['Youtuber'], bar_colms['subscribers (MM)'],
                    color=sns.color_palette('hls', len(bar_colms)))

    for i in range(len(bars)):
        plt.text(
            bars[i].get_width() + 0.1,
            bars[i].get_y() + bars[i].get_height() / 2,
            f'{bar_colms["subscribers (MM)"].iloc[i]}M',
            va='center',
            fontsize=12,
            weight='bold'
        )

    plt.xlabel("No. of Subscribers in Million", weight='bold', fontsize=12)
    plt.ylabel("Youtuber", weight='bold', fontsize=12)
    plt.title("Top 10 YouTube Channels by Subscribers", weight='bold', fontsize=14)
    plt.show()


def plot_top_youtube_channels_by_views(df):
    """绘制前10个YouTube频道的视频观看次数条形图"""
    colms = ['Youtuber', 'video views']
    bar_colms = df.loc[0:9, colms].sort_values('video views', ascending=True)
    bar_colms['video views (bil)'] = (bar_colms['video views'] / 1000000000).astype(int)

    fig = plt.figure(figsize=(10, 6))
    bars = plt.barh(bar_colms['Youtuber'], bar_colms['video views (bil)'],
                    color=sns.color_palette('hls', len(bar_colms)))

    for i in range(len(bars)):
        plt.text(
            bars[i].get_width() + 0.05,
            bars[i].get_y() + bars[i].get_height() / 2,
            f'{bar_colms["video views (bil)"].iloc[i]}B',
            va='center',
            fontsize=12,
            weight='bold'
        )

    plt.xlabel("Total Video Views in Billion", weight='bold', fontsize=12)
    plt.ylabel("Youtuber", weight='bold', fontsize=12)
    plt.title("Top 10 YouTube Channels by Total Video Views", weight='bold', fontsize=14)
    plt.show()


def plot_relationship_scatter_subscribers_vs_views(df):
    """绘制订阅者数量与视频观看次数的关系散点图"""
    plot_relationship_scatter(
        df, 'subscribers', 'video views',
        'Relationship Between Number of Subscribers & Total Video Views',
        'Subscribers (Millions)', 'Total Video Views (Billions)'
    )


def plot_relationship_scatter_views_vs_earnings(df):
    """绘制视频观看次数与最高年度收入的关系散点图"""
    plot_relationship_scatter(
        df, 'video views', 'highest_yearly_earnings',
        'Relationship Between Total Video Views & Highest Yearly Earning',
        'Total Video Views (Billions)', 'Highest Yearly Earning (Millions)'
    )


def plot_relationship_scatter_views_vs_uploads(df):
    """绘制视频观看次数与上传视频总数的关系散点图"""
    # 选择相关列
    colms = ['uploads', 'video views']

    # 提取相关数据
    scatter_colms = df.loc[:, colms]

    # 转换为更适合的单位
    scatter_colms['video views (bil)'] = (scatter_colms['video views'] / 1000000000).astype(int)

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


def plot_relationship_scatter(df, x_col, y_col, title, x_label, y_label, size_col=None, cmap='plasma', alpha=0.8):
    """
    绘制关系散点图

    参数:
    df (DataFrame): 包含数据的数据框
    x_col (str): x轴的列名
    y_col (str): y轴的列名
    title (str): 图表的标题
    x_label (str): x轴的标签
    y_label (str): y轴的标签
    size_col (str, 可选): 控制点大小的列名，默认为None
    cmap (str): 颜色映射，默认为'plasma'
    alpha (float): 点的透明度，默认为0.8

    返回:
    无直接返回值，但显示一个散点图。
    """

    # 复制x_col和y_col列的数据以避免对原始数据的修改
    scatter_cols = df[[x_col, y_col]].copy()

    # 根据列名包含的关键词对数据进行单位转换，使数据更具可读性
    if 'subscribers' in x_col:
        scatter_cols[x_col] = (scatter_cols[x_col] / 1000000).astype(int)  # 将订阅者数转换为百万
    else:
        scatter_cols[x_col] = (scatter_cols[x_col] / 1000000000).astype(int)  # 将其他数值转换为十亿

    if 'video views' in y_col:
        scatter_cols[y_col] = (scatter_cols[y_col] / 1000000000).astype(int)  # 将视频观看次数转换为十亿
    else:
        scatter_cols[y_col] = (scatter_cols[y_col] / 1000000).astype(int)  # 将其他数值转换为百万

    # 创建图表对象，并设置图表大小
    fig = plt.figure(figsize=(10, 6))

    # 如果提供了控制点大小的列名，则根据该列设置点的大小
    size = scatter_cols[size_col] * 0.5 if size_col else None

    # 绘制散点图
    plt.scatter(
        scatter_cols[x_col],  # x轴数据
        scatter_cols[y_col],  # y轴数据
        s=size,  # 点的大小
        c=scatter_cols[y_col],  # 点的颜色，根据y轴数据进行着色
        cmap=cmap,  # 颜色映射
        alpha=alpha,  # 点的透明度
        edgecolors="w",  # 点的边缘颜色
        linewidth=0.5  # 点的边缘宽度
    )

    # 设置图表的标题和轴标签
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(x_label, fontsize=12, weight='bold')
    plt.ylabel(y_label, fontsize=12, weight='bold')

    # 添加颜色条，并设置其标签
    plt.colorbar(label=y_label)

    # 显示图表
    plt.show()


if __name__ == "__main__":
    setup_plotting_style()

    # 加载和预处理数据
    df = load_and_preprocess_data(CONFIG['input_file'], CONFIG['encoding'])

    # 保存清理后的数据
    save_cleaned_data(df, CONFIG['output_file'], CONFIG['encoding'])

    # 可视化分析
    plot_correlation_heatmap(df)
    plot_channels_created_per_year(df)
    plot_channel_type_distribution(df)
    plot_country_distribution(df)
    plot_top_youtube_channels_by_subscribers(df)
    plot_top_youtube_channels_by_views(df)
    plot_relationship_scatter_subscribers_vs_views(df)
    plot_relationship_scatter_views_vs_earnings(df)
    plot_relationship_scatter_views_vs_uploads(df)