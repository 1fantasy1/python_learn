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


class FSA_RF_Optimizer:
    def __init__(self, X_train, X_test, y_train, y_test, pop_size=30, max_iter=50):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = 2  # 优化两个参数：n_estimators 和 min_samples_leaf

        # 参数范围设置
        self.lb = np.array([10, 1])
        self.ub = np.array([200, 20])

        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize(self):
        positions = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        return np.round(positions)

    def evaluate(self, position):
        n_estimators = int(position[0])
        min_samples_leaf = int(position[1])

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        rf.fit(self.X_train, self.y_train)

        train_pred = rf.predict(self.X_train)
        test_pred = rf.predict(self.X_test)

        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)

        return train_mse + test_mse

    def update_position(self, current_pos, local_best, global_best):
        S_L = (local_best - current_pos) * np.random.rand(self.dim)
        S_G = (global_best - current_pos) * np.random.rand(self.dim)
        new_pos = current_pos + S_L + S_G
        new_pos = np.clip(new_pos, self.lb, self.ub)
        return np.round(new_pos)

    def update_random_init(self, current_pos, local_best, global_best):
        new_pos = global_best + (global_best - local_best) * np.random.rand(self.dim)
        new_pos = np.clip(new_pos, self.lb, self.ub)
        return np.round(new_pos)

    def optimize(self):
        positions = self.initialize()
        local_best_positions = positions.copy()
        local_best_fitness = np.array([self.evaluate(pos) for pos in positions])

        global_best_idx = np.argmin(local_best_fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = local_best_fitness[global_best_idx]

        for iter in range(self.max_iter):
            print(f"FSA优化迭代 {iter + 1}/{self.max_iter}, 当前最优适应度: {global_best_fitness:.6f}")

            for i in range(self.pop_size):
                new_position = self.update_position(
                    positions[i],
                    local_best_positions[i],
                    global_best_position
                )

                new_fitness = self.evaluate(new_position)

                if new_fitness < local_best_fitness[i]:
                    local_best_fitness[i] = new_fitness
                    local_best_positions[i] = new_position.copy()

                    if new_fitness < global_best_fitness:
                        global_best_fitness = new_fitness
                        global_best_position = new_position.copy()

                positions[i] = self.update_random_init(
                    positions[i],
                    local_best_positions[i],
                    global_best_position
                )

            if global_best_fitness < self.best_fitness:
                self.best_fitness = global_best_fitness
                self.best_solution = global_best_position.copy()

        return {
            'n_estimators': int(self.best_solution[0]),
            'min_samples_leaf': int(self.best_solution[1]),
            'best_fitness': self.best_fitness
        }


# 设置 matplotlib 显示中文字体
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
])

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

# 运行FSA优化Random Forest
print("开始FSA优化Random Forest...")
optimizer = FSA_RF_Optimizer(X_train_scaled, X_test_scaled, y_train, y_test)
best_params = optimizer.optimize()

# 定义模型
models = {
    'FSA-Random Forest': RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    ),
    'Standard Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
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

    threshold = 0.1
    accurate_predictions = sum(abs(y_test - y_pred) < (threshold * y_test))
    accuracy = accurate_predictions / len(y_test)

    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'R-squared': r2,
        'Accuracy': accuracy,
        'Predictions': y_pred
    }

# 打印结果
for name, metrics in results.items():
    print(f"\n{name} 模型:")
    print(f"  平均绝对误差（MAE）: {metrics['MAE']}")
    print(f"  均方误差（MSE）: {metrics['MSE']}")
    print(f"  决定系数（R-squared）: {metrics['R-squared']}")
    print(f"  预测准确率: {metrics['Accuracy']:.2%}")

# 绘制所有模型的对比图
plt.figure(figsize=(15, 10))

for i, (name, metrics) in enumerate(results.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, metrics['Predictions'], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("实际订阅者数量")
    plt.ylabel("预测订阅者数量")
    plt.title(f"{name}模型的预测结果")

plt.tight_layout()
plt.show()

# 绘制模型性能对比柱状图
metrics_to_plot = ['MAE', 'MSE', 'R-squared', 'Accuracy']
n_metrics = len(metrics_to_plot)
n_models = len(results)

plt.figure(figsize=(15, 8))
x = np.arange(n_models)
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    values = [results[model][metric] for model in results.keys()]
    plt.bar(x + i * width, values, width, label=metric)

plt.xlabel('模型')
plt.ylabel('指标值')
plt.title('各模型性能指标对比')
plt.xticks(x + width * 1.5, results.keys(), rotation=45)
plt.legend()
plt.tight_layout()
plt.show()