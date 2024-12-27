from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 日志设置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 1. 数据预处理函数
def preprocess_data(df):
    def convert_to_int(value):
        if isinstance(value, str):
            value = value.replace(',', '').replace('M', 'e6').replace('B', 'e9')
            return int(float(value))
        return value

    df['Subscribers'] = df['Subscribers'].apply(convert_to_int)
    df['Uploads'] = df['Uploads'].apply(convert_to_int)
    df['Views'] = df['Views'].apply(convert_to_int)
    df = df.dropna()
    df = pd.get_dummies(df, columns=['Country'], dummy_na=True)
    return df


# 2. 特征工程函数
def feature_engineering(df):
    df['Subscribers_per_Upload'] = df['Subscribers'] / (df['Uploads'] + 1)
    df['Views_per_Subscriber'] = df['Views'] / (df['Subscribers'] + 1)
    return df


# 3. 模型训练与评估函数
def train_and_evaluate_model(X, y):
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 模型训练
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 模型预测
    y_pred = model.predict(X_test_scaled)

    # 模型评估
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"平均绝对误差（MAE）：{mae}")
    logging.info(f"均方误差（MSE）：{mse}")
    logging.info(f"决定系数（R²）：{r2}")

    return model, y_test, y_pred


# 4. 数据可视化函数
def visualize_data(df):
    # 检查是否存在 'Country' 列
    if 'Country' not in df.columns:
        logging.warning("数据集中没有 'Country' 列，将跳过饼图绘制。")
    else:
        # 按国家分布的频道饼图
        plt.figure(figsize=(8, 8))
        country_counts = df['Country'].dropna().value_counts()
        explode = [0.1 if count == max(country_counts) else 0 for count in country_counts]
        country_counts.plot.pie(autopct='%1.1f%%', startangle=140, cmap='Set3', explode=explode)
        plt.title("按国家分布的频道", fontsize=16)
        plt.ylabel("")  # 隐藏y轴标签
        plt.tight_layout()
        plt.show()

    # 前 10 个订阅者最多的频道柱状图
    plt.figure(figsize=(10, 6))
    top_channels = df.nlargest(10, 'Subscribers')
    sns.barplot(
        data=top_channels,
        x='Subscribers',
        y='Username',
        hue='Username',
        palette='coolwarm',
        dodge=False
    )
    plt.title("前10个订阅者最多的频道", fontsize=16)
    plt.xlabel("订阅者数量", fontsize=12)
    plt.ylabel("频道名称", fontsize=12)
    plt.legend(title="频道", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



# 5. 模型结果可视化函数
def plot_results(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='预测值')
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='理想线')
    plt.xlabel('实际订阅者数量', fontsize=12)
    plt.ylabel('预测订阅者数量', fontsize=12)
    plt.title('实际值与预测值对比', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 数据加载
    df = pd.read_csv("YOUTUBE CHANNELS DATASET.csv")

    # 数据预处理
    df = preprocess_data(df)

    # 特征工程
    df = feature_engineering(df)

    # 可视化数据分布
    visualize_data(df)

    # 特征与目标变量
    features = df[['Ranking', 'Uploads', 'Views'] + [col for col in df.columns if 'Country' in col]]
    target = df['Subscribers']

    # 模型训练与评估
    model, y_test, y_pred = train_and_evaluate_model(features, target)

    # 可视化模型结果
    plot_results(y_test, y_pred)
