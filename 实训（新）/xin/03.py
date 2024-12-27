import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from matplotlib.ticker import ScalarFormatter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# 设置 matplotlib 显示中文字体，防止出现乱码
rcParams['font.sans-serif'] = ['SimHei']


# 数据读取函数
def read_data():
    """
    读取YouTube数据集
    """
    # return pd.read_csv("YOUTUBE CHANNELS DATASET.csv")
    return pd.read_csv("001.csv", encoding="latin-1")

# 数据预处理函数，将带有单位或逗号分隔的数字转换为整数类型
def convert_to_int(value):
    if isinstance(value, str):  # 如果值是字符串类型
        # 替换字符串中的逗号，"M" 表示百万（1e6），"B" 表示十亿（1e9）
        value = value.replace(',', '').replace('M', 'e6').replace('B', 'e9')
        return int(float(value))  # 将字符串转换为浮点数后再转为整数
    return value  # 如果值不是字符串，则直接返回原值


# 数据预处理主函数，对指定列应用转换函数
def preprocess_data(df):
    """
    对数据集中的 'Subscribers'、'Uploads'、'Views' 列进行预处理，转换为整数类型
    """
    columns_to_process = ['subscribers', 'uploads', 'video views']
    for col in columns_to_process:
        df[col] = df[col].apply(convert_to_int)
    return df


# 可视化前10个订阅者最多的频道的函数
def visualize_top_subscribers(df):
    """
    绘制柱状图展示前10个订阅者最多的频道
    """
    plt.figure(figsize=(10, 6))
    top_channels = df.nlargest(10, 'subscribers')
    sns.barplot(data=top_channels, x='subscribers', y='Username', hue='Username', palette='coolwarm', dodge=False)
    plt.title("前10个订阅者最多的频道", fontsize=16)
    plt.xlabel("订阅者（以百万计）", fontsize=12)
    plt.ylabel("频道名称", fontsize=12)
    plt.tight_layout()
    plt.show()


# 绘制按国家分布的频道饼图的函数
def visualize_country_distribution(df):
    """
    绘制饼图显示按国家分布的频道
    """
    plt.figure(figsize=(8, 8))
    country_counts = df['Country'].dropna().value_counts()
    # 设置爆炸效果，仅对最大值进行爆炸
    explode = [0.1 if count == max(country_counts) else 0 for count in country_counts]
    # 绘制饼图
    country_counts.plot.pie(autopct='%1.1f%%', startangle=140, cmap='Set3', explode=explode)
    plt.title("按国家分布的频道", fontsize=16)
    plt.ylabel("")  # 隐藏y轴标签
    plt.tight_layout()
    plt.show()


# 绘制各国上传视频数量与总观看量关系的散点图函数
def visualize_uploads_views_relationship(df):
    """
    绘制散点图显示各国上传视频数量与总观看量的关系
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Uploads', y='Views', hue='Country', style='Country', palette='tab10', s=30)
    plt.title("各国上传视频数量与总观看量的关系", fontsize=16)
    plt.xlabel("上传视频数量", fontsize=12)
    plt.ylabel("总观看量（以亿计）", fontsize=12)
    plt.legend(title="国家", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 使用科学计数法显示纵轴
    ax = plt.gca()  # 获取当前的轴
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

# 数据清洗函数，删除包含缺失值的行，并进行独热编码处理
def clean_and_encode_data(df):
    """
    删除包含缺失值的行，对 'Country' 列进行独热编码处理（包括缺失值的处理）
    """
    df = df.dropna()
    df = pd.get_dummies(df, columns=['Country'], dummy_na=True)
    return df


# 构建特征和目标变量的函数
def build_features_target(df):
    """
    构建特征和目标变量
    """
    features = df[['Uploads', 'Views'] + [col for col in df.columns if 'Country_' in col]]
    target = df['subscribers']
    return features, target


# 模型训练与预测函数
def train_and_predict(X, y):
    """
    划分训练集和测试集，训练随机森林回归模型，进行预测并返回预测结果
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test


# 特征标准化函数
def scale_features(X_train, X_test):
    """
    对特征进行标准化处理
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 计算准确率函数
def calculate_accuracy(y_test, y_pred):
    """
    计算预测准确率 (1 - MAPE)
    """
    mape = np.mean(np.abs((y_test - y_pred) / y_test))  # 计算平均绝对百分比误差
    accuracy = 1 - mape  # 准确率
    return accuracy

# 评估指标计算与输出函数
def calculate_and_print_metrics_with_accuracy(y_test, y_pred):
    """
    计算平均绝对误差、均方误差、R-squared（决定系数）以及预测准确率，并输出
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"R-squared（决定系数）：{r2}")
    print(f"预测准确率（Accuracy）：{accuracy * 100:.2f}%")


# 绘制实际值与预测值对比图的函数
def visualize_predictions(y_test, y_pred, name):
    """
    绘制实际值与预测值的对比图
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='预测值')
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='理想线')  # 理想线（实际值=预测值）
    plt.xlabel('实际订阅者数量', fontsize=12)
    plt.ylabel('预测订阅者数量', fontsize=12)
    plt.title(f'{name}实际值与预测值对比', fontsize=16)
    plt.legend()

    # 设置纵轴使用科学计数法显示
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()

# 决策树模型训练与预测函数
def train_and_predict_decision_tree(X, y):
    """
    划分训练集和测试集，训练决策树回归模型，进行预测并返回预测结果
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test

# K 近邻模型训练与预测函数
def train_and_predict_knn(X, y, n_neighbors=5):
    """
    划分训练集和测试集，训练 K 近邻回归模型，进行预测并返回预测结果
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test

# 支持向量机模型训练与预测函数
def train_and_predict_svm(X, y):
    """
    划分训练集和测试集，训练支持向量机回归模型，进行预测并返回预测结果
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVR(kernel='rbf')  # 使用径向基核函数
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test

if __name__ == "__main__":
    # 读取数据
    data = read_data()
    # 数据预处理
    processed_data = preprocess_data(data)
    # 可视化分析
    visualize_top_subscribers(processed_data)
    visualize_country_distribution(processed_data)
    visualize_uploads_views_relationship(processed_data)
    # 数据清洗与编码
    cleaned_data = clean_and_encode_data(processed_data)
    # 构建特征和目标变量
    features, target = build_features_target(cleaned_data)
    # 特征标准化
    X_train_scaled, X_test_scaled = scale_features(features, features)

    # 随机森林模型训练与评估
    print("随机森林回归模型评估结果：")
    y_pred_rf, y_test_rf = train_and_predict(X_train_scaled, target)
    calculate_and_print_metrics_with_accuracy(y_test_rf, y_pred_rf)
    visualize_predictions(y_test_rf, y_pred_rf, '随机森林回归')

    # 决策树模型训练与评估
    print("决策树回归模型评估结果：")
    y_pred_dt, y_test_dt = train_and_predict_decision_tree(X_train_scaled, target)
    calculate_and_print_metrics_with_accuracy(y_test_rf, y_pred_rf)
    visualize_predictions(y_test_dt, y_pred_dt, '决策树')

    # K 近邻模型训练与评估
    print("K 近邻回归模型评估结果：")
    y_pred_knn, y_test_knn = train_and_predict_knn(X_train_scaled, target)
    calculate_and_print_metrics_with_accuracy(y_test_knn, y_pred_knn)
    visualize_predictions(y_test_knn, y_pred_knn, 'K 近邻')

    # 支持向量机模型训练与评估
    print("支持向量机回归模型评估结果：")
    y_pred_svm, y_test_svm = train_and_predict_svm(X_train_scaled, target)
    calculate_and_print_metrics_with_accuracy(y_test_svm, y_pred_svm)
    visualize_predictions(y_test_svm, y_pred_svm, '支持向量机')