import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

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

# 设置matplotlib显示中文字体
plt.rcParams['font.sans-serif'] = [CONFIG['font_family']]
plt.rcParams['axes.unicode_minus'] = False


def setup_plotting_style():
    try:
        plt.style.use(CONFIG['style'])
    except OSError:
        plt.style.use('ggplot')


def load_data(file_path, encoding='utf-8'):
    """加载CSV文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, encoding=encoding)


def preprocess_data(df):
    """数据预处理"""
    # 删除不必要的列
    columns_to_drop = [
        'rank', 'Abbreviation', 'country_rank', 'created_month',
        'created_date', 'Gross tertiary education enrollment (%)',
        'Unemployment rate', 'Urban_population'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 处理缺失值
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['int64', 'float']).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df[num_cols] = df[num_cols].fillna(0)

    # 转换数据类型
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
    df = df.astype(type_conversions)

    # 移除视频观看次数为0的行
    df = df[df['video views'] > 0]

    # 按订阅者数量降序排列并重置索引
    df.sort_values(by='subscribers', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def save_cleaned_data(df, file_path, encoding='utf-8'):
    """保存清理后的数据"""
    df.to_csv(file_path, index=False, encoding=encoding)
    print(f"Data has been successfully saved to {file_path}")


def plot_correlation_matrix(df):
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
    channels_in_year = channels_in_year[
        (channels_in_year['Year'] > 1970) & (channels_in_year['Year'] <= 2022)].sort_values(by='Year')

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


def plot_top_youtube_channels(df, column, title, unit_label, conversion_factor, sort_column='subscribers', top_n=10):
    """通用的绘制前N个YouTube频道条形图函数"""
    bar_cols = df.loc[0:top_n - 1, ['Youtuber', sort_column]].sort_values(sort_column, ascending=True)
    bar_cols[f'{column} ({unit_label})'] = (bar_cols[sort_column] / conversion_factor).astype(int)

    fig = plt.figure(figsize=(10, 6))
    bars = plt.barh(bar_cols['Youtuber'], bar_cols[f'{column} ({unit_label})'],
                    color=sns.color_palette('hls', len(bar_cols)))

    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f'{bar_cols[f"{column} ({unit_label})"].iloc[i]}{unit_label}',
            va='center',
            fontsize=12,
            weight='bold'
        )

    plt.xlabel(f"No. of {title} in {unit_label}", weight='bold', fontsize=12)
    plt.ylabel("Youtuber", weight='bold', fontsize=12)
    plt.title(f"Top {top_n} YouTube Channels by {title}", weight='bold', fontsize=14)
    plt.show()


def plot_relationship_scatter(df, x_col, y_col, title, x_label, y_label, size_col=None, cmap='plasma', alpha=0.8):
    """绘制关系散点图"""
    scatter_cols = df[[x_col, y_col]].copy()
    scatter_cols[x_col] = (scatter_cols[x_col] / 1000000).astype(int) if 'subscribers' in x_col else (
                scatter_cols[x_col] / 1000000000).astype(int)
    scatter_cols[y_col] = (scatter_cols[y_col] / 1000000000).astype(int) if 'video views' in y_col else (
                scatter_cols[y_col] / 1000000).astype(int)

    fig = plt.figure(figsize=(10, 6))
    size = scatter_cols[size_col] * 0.5 if size_col else None
    plt.scatter(scatter_cols[x_col], scatter_cols[y_col], s=size, c=scatter_cols[y_col], cmap=cmap, alpha=alpha,
                edgecolors="w", linewidth=0.5)

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(x_label, fontsize=12, weight='bold')
    plt.ylabel(y_label, fontsize=12, weight='bold')
    plt.colorbar(label=y_label)
    plt.show()


if __name__ == "__main__":
    setup_plotting_style()

    # 加载数据
    df = load_data(CONFIG['input_file'], CONFIG['encoding'])

    # 数据预处理
    df = preprocess_data(df)

    # 保存清理后的数据
    save_cleaned_data(df, CONFIG['output_file'], CONFIG['encoding'])

    # 可视化分析
    plot_correlation_matrix(df)
    plot_channels_created_per_year(df)
    plot_channel_type_distribution(df)
    plot_country_distribution(df)
    plot_top_youtube_channels(df, 'Subscribers', 'Subscribers', 'M', 1000000, sort_column='subscribers')
    plot_top_youtube_channels(df, 'Video Views', 'Total Video Views', 'B', 1000000000, sort_column='video views')
    plot_relationship_scatter(df, 'subscribers', 'video views', 'Relationship Between Number of Subscribers & Total Video Views', 'Subscribers (Millions)', 'Total Video Views (Billions)')