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
    'dpi': 1000,
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
    columns_to_drop = [
        'rank', 'Abbreviation', 'country_rank', 'created_month',
        'created_date', 'Gross tertiary education enrollment (%)',
        'Unemployment rate', 'Urban_population'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    fill_missing_values(df)
    convert_data_types(df)
    df = df[df['video views'] > 0]
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
    print(f"数据已成功保存到 {file_path}")


def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
    numeric_df = df.select_dtypes(include=[float, int])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=CONFIG['figure_size'])
    sns.heatmap(correlation_matrix, cmap="BuPu", annot=True)
    plt.title("相关性热力图", fontsize=18, weight='bold')
    plt.show()


def plot_channels_created_per_year(df):
    """统计每年创建频道数并绘图"""
    channels_in_year = df['created_year'].value_counts().reset_index()
    channels_in_year.columns = ['年份', '创建频道数']
    channels_in_year = channels_in_year[(channels_in_year['年份'] > 1970) & (channels_in_year['年份'] <= 2022)]
    channels_in_year.sort_values(by='年份', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(channels_in_year['年份'], channels_in_year['创建频道数'], color='#32A645', label='频道数量（柱状图）')
    ax.plot(channels_in_year['年份'], channels_in_year['创建频道数'], color='#004DFF', marker='o',
            label='频道数量（折线图）')
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("频道数量", fontsize=12)
    ax.set_title("每年创建的频道数量", fontsize=14, weight='bold')
    ax.legend()
    plt.show()


def plot_channel_type_distribution(df):
    """绘制频道类型分布饼图"""
    channel_types = df['channel_type'].value_counts()
    total_count = channel_types.sum()
    filtered_channel_types = channel_types[channel_types / total_count >= 0.005]
    others_count = channel_types[channel_types / total_count < 0.005].sum()
    filtered_channel_types['其他'] = others_count

    plt.figure(figsize=(8, 8))
    plt.pie(
        filtered_channel_types,
        labels=filtered_channel_types.index,
        autopct=lambda p: f'{p:.1f}% ({int(p * total_count / 100)})',
        colors=sns.color_palette('hls', len(filtered_channel_types))
    )
    plt.title('YouTube频道类型分布', weight='bold', fontsize=14)
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
    plt.title('YouTube频道按国家分布（前15名）', weight='bold', fontsize=14)
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

    plt.xlabel("订阅者数量（百万）", weight='bold', fontsize=12)
    plt.ylabel("YouTuber", weight='bold', fontsize=12)
    plt.title("前10名YouTube频道的订阅者", weight='bold', fontsize=14)
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

    plt.xlabel("总视频观看次数（十亿）", weight='bold', fontsize=12)
    plt.ylabel("YouTuber", weight='bold', fontsize=12)
    plt.title("前10名YouTube频道的总视频观看次数", weight='bold', fontsize=14)
    plt.show()


def plot_relationship_scatter_subscribers_vs_views(df):
    """绘制订阅者数量与视频观看次数的关系散点图"""
    plot_relationship_scatter(
        df, 'subscribers', 'video views',
        '订阅者数量与总视频观看次数的关系',
        '订阅者数量（百万）', '总视频观看次数（十亿）'
    )


def plot_relationship_scatter_views_vs_earnings(df):
    """绘制视频观看次数与最高年度收入的关系散点图"""
    plot_relationship_scatter(
        df, 'video views', 'highest_yearly_earnings',
        '总视频观看次数与最高年度收入的关系',
        '总视频观看次数（十亿）', '最高年度收入（百万）'
    )


def plot_relationship_scatter_views_vs_uploads(df):
    """绘制视频观看次数与上传视频总数的关系散点图"""
    colms = ['uploads', 'video views']
    scatter_colms = df.loc[:, colms]
    scatter_colms['video views (bil)'] = (scatter_colms['video views'] / 1000000000).astype(int)
    x = scatter_colms['video views (bil)']
    y = scatter_colms['uploads']
    size = scatter_colms['video views (bil)'] * 0.6

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=size, c=x, cmap='cool_r', alpha=0.9, edgecolors="w", linewidth=0.5)
    plt.title('总视频观看次数与总上传视频数的关系', fontsize=14, weight='bold')
    plt.xlabel('总视频观看次数（十亿）', fontsize=12, weight='bold')
    plt.ylabel('总上传视频数', fontsize=12, weight='bold')
    plt.colorbar(label='总视频观看次数（十亿）')
    plt.show()


def plot_relationship_scatter(df, x_col, y_col, title, x_label, y_label, size_col=None, cmap='plasma', alpha=0.8):
    """绘制关系散点图"""
    scatter_cols = df[[x_col, y_col]].copy()
    if 'subscribers' in x_col:
        scatter_cols[x_col] = (scatter_cols[x_col] / 1000000).astype(int)
    else:
        scatter_cols[x_col] = (scatter_cols[x_col] / 1000000000).astype(int)
    if 'video views' in y_col:
        scatter_cols[y_col] = (scatter_cols[y_col] / 1000000000).astype(int)
    else:
        scatter_cols[y_col] = (scatter_cols[y_col] / 1000000).astype(int)

    fig = plt.figure(figsize=(10, 6))
    size = scatter_cols[size_col] * 0.5 if size_col else None

    plt.scatter(
        scatter_cols[x_col],
        scatter_cols[y_col],
        s=size,
        c=scatter_cols[y_col],
        cmap=cmap,
        alpha=alpha,
        edgecolors="w",
        linewidth=0.5
    )
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(x_label, fontsize=12, weight='bold')
    plt.ylabel(y_label, fontsize=12, weight='bold')
    plt.colorbar(label=y_label)
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