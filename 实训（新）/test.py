import numpy as np
import pandas as pd
import math
import torch
import numpy as np
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# 读取并预处理数据（与之前类似）
df = pd.read_csv('yahoo_details_5_years.csv')

# 筛选需要的公司数据，例如AAPL
df = df[df['Company'] == "AAPL"].copy()

# 转换日期格式并设置为索引
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)

# 选择需要的列并处理缺失值
df = df[['Close']].copy()
df.dropna(inplace=True)

# 归一化数据
scaler = MinMaxScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# 划分训练集和测试集
train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]


# 创建序列数据
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out)])
    return np.array(X), np.array(y)


n_steps_in, n_steps_out = 5, 1
X_train, y_train = create_sequences(train_df['Close_scaled'].values, n_steps_in, n_steps_out)
X_test, y_test = create_sequences(test_df['Close_scaled'].values, n_steps_in, n_steps_out)

# 转换为 PyTorch 的张量
X_train = torch.from_numpy(X_train).float().unsqueeze(-1)
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float().unsqueeze(-1)
y_test = torch.from_numpy(y_test).float()

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, num_layers=4, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型、损失函数和优化器
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 准备数据
# ...（与之前相同的数据预处理和张量转换）

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 如果需要，可以添加验证过程
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

# 测试模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# 反归一化预测结果
y_pred_inverse = scaler.inverse_transform(y_pred.numpy())
y_test_inverse = scaler.inverse_transform(y_test.numpy())

# 计算评价指标
rmse = math.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
r2 = r2_score(y_test_inverse, y_pred_inverse)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R2 Score: {r2:.2f}")

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(test_df.index[n_steps_in:], y_test_inverse.flatten(), label='Actual')
plt.plot(test_df.index[n_steps_in:], y_pred_inverse.flatten(), label='Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()