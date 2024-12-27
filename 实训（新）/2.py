import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']
import warnings
warnings.filterwarnings('ignore')

# 加载数据集
file_path = 'retail_store_inventory.csv'
df = pd.read_csv(file_path)

# 将 'Date' 列转换为日期时间格式
df['Date'] = pd.to_datetime(df['Date'])

# 处理缺失值（这里以填充中位数为例）
df.fillna(df.median(), inplace=True)

# 对分类变量进行独热编码
df = pd.get_dummies(df, columns=['Category'], drop_first=True)

# 选择特征和目标变量
features = df.drop(columns=['Units Sold'])
target = df['Units Sold']

# 标准化数值型特征（可选）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_features = features.select_dtypes(include=[np.number]).columns
features[numeric_features] = scaler.fit_transform(features[numeric_features])

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 构建神经网络模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 定义模型结构
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # 输入层和第一隐层
model.add(Dense(32, activation='relu'))  # 第二隐层
model.add(Dense(1))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# 在测试集上进行预测
y_pred = model.predict(X_test).flatten()

# 计算均方误差 (MSE) 和均方根误差 (RMSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('测试集 RMSE:', rmse)

# 可视化训练过程中的损失下降
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('迭代周期')
plt.legend()
plt.show()

# 可视化实际值与预测值的关系
plt.scatter(y_test, y_pred)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs. 预测值')
plt.show()