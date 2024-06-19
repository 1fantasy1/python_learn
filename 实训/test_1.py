import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 数据加载
url = 'https://archive.ics.uci.edu/static/public/109/data.csv'
df = pd.read_csv(url)

# 预览数据
print(df.head())

# 检查是否有缺失值
print(df.isnull().sum())

# 对所有特征进行z-score标准化
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])

# 显示标准化后的数据
print(df_scaled.head())

# 对'Alcohol'特征进行等宽分箱
num_bins = 5
df['Alcohol_bin'] = pd.cut(df['Alcohol'], bins=num_bins, labels=False)

# 显示分箱后的数据
print(df[['Alcohol', 'Alcohol_bin']].head())
