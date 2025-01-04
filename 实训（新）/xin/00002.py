import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 显示中文字体，防止出现乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
file_path = "002.csv"
df = pd.read_csv(file_path, encoding="latin-1")

# 数据预处理
df = df.dropna(subset=[
    'subscribers', 'video views', 'uploads', 'channel_type_rank',
    'video_views_rank', 'video_views_for_the_last_30_days',
    'highest_monthly_earnings', 'highest_yearly_earnings', 'created_year'
])  # 去除缺失值行

# 特征选择
features = df[[
    'video views', 'uploads', 'channel_type_rank', 'video_views_rank',
    'video_views_for_the_last_30_days', 'highest_monthly_earnings',
    'highest_yearly_earnings', 'created_year'
]]
target = df['subscribers']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Machine': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Naive Bayes': GaussianNB()
}

# 存储结果
results = {}

# 训练和评估每个模型
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 预测准确率
    threshold = 0.1  # 误差阈值，定义为实际值的10%
    accurate_predictions = sum(abs(y_test - y_pred) < (threshold * y_test))
    accuracy = accurate_predictions / len(y_test)

    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'R-squared': r2,
        'Accuracy': accuracy
    }

# 打印结果
for name, metrics in results.items():
    print(f"{name} 模型:")
    print(f"  平均绝对误差（MAE）: {metrics['MAE']}")
    print(f"  均方误差（MSE）: {metrics['MSE']}")
    print(f"  决定系数（R-squared）: {metrics['R-squared']}")
    print(f"  预测准确率: {metrics['Accuracy']:.2%}")
    print()

# 绘制实际值与预测值的对比图（以随机森林为例）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, models['Random Forest'].predict(X_test_scaled), alpha=0.6)
plt.xlabel("实际订阅者数量")
plt.ylabel("预测订阅者数量")
plt.title("随机森林模型的实际值与预测值对比")
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 假设你已经准备好训练集 X_train, y_train 和 测试集 X_test, y_test

# 初始化随机森林回归器
model = RandomForestRegressor()

# 训练模型
model.fit(X_train, y_train)

# 使用测试集进行预测
test_predictions = model.predict(X_test)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_test, test_predictions)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, test_predictions)

# 计算决定系数（R-squared）
r2 = r2_score(y_test, test_predictions)

# 计算预测准确率（可以理解为决定系数的百分比）
accuracy = r2 * 100

# 输出结果
print(f"FSA-RF模型:")
print(f"平均绝对误差（MAE）: {mae}")
print(f"均方误差（MSE）: {mse}")
print(f"决定系数（R-squared）: {r2}")
print(f"预测准确率: {accuracy:.2f}%")
