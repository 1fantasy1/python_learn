import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
data = pd.read_csv("2.csv")

# 特征和目标变量
features = ['video_views', 'uploads', 'video_views_for_the_last_30_days', 'subscribers_for_last_30_days']
target = 'subscribers'

# 分割数据集
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)