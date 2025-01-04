# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
# # 设置 matplotlib 显示中文字体，防止出现乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 加载数据
# file_path = "002.csv"
# df = pd.read_csv(file_path, encoding="latin-1")
#
# # 数据预处理
# df = df.dropna(subset=['subscribers', 'video views', 'uploads'])  # 去除缺失值行
#
# # 特征选择
# features = df[['video views', 'uploads']]
# target = df['subscribers']
#
# # 数据分割
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
# # 特征标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 模型训练
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train_scaled, y_train)
#
# # 预测
# y_pred = model.predict(X_test_scaled)
#
# # 评估
# mse = mean_squared_error(y_test, y_pred)
# print(f"均方误差: {mse}")
#
# # 绘制预测结果与实际值的对比
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.6)
# plt.xlabel("实际订阅者数量")
# plt.ylabel("预测订阅者数量")
# plt.title("实际值与预测值的对比")
# plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

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

# 模型训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评估
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"平均绝对误差（MAE）: {mae}")
print(f"均方误差（MSE）: {mse}")
print(f"决定系数（R-squared）: {r2}")

# 绘制预测结果与实际值的对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("实际订阅者数量")
plt.ylabel("预测订阅者数量")
plt.title("实际值与预测值的对比")
plt.show()