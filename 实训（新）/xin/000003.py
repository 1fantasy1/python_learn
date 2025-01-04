import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import cross_val_score

# 设置 matplotlib 显示中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class FSA_RF_Optimizer:
    def __init__(self, X, y, pop_size=30, max_iter=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 数据集分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 转换为PyTorch张量
        if torch.cuda.is_available():
            self.X_train = torch.FloatTensor(self.X_train).to(self.device)
            self.X_test = torch.FloatTensor(self.X_test).to(self.device)
            self.y_train = torch.FloatTensor(self.y_train).to(self.device)
            self.y_test = torch.FloatTensor(self.y_test).to(self.device)

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = 2
        self.lb = np.array([10, 1])
        self.ub = np.array([200, 20])
        self.best_solution = None
        self.best_fitness = float('inf')

    def evaluate(self, position):
        n_estimators = int(position[0])
        min_samples_leaf = int(position[1])

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # 使用交叉验证评估模型
        scores = cross_val_score(rf, self.X_train.cpu().numpy(),
                                 self.y_train.cpu().numpy(),
                                 cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)  # 返回负MSE的平均值

    def optimize(self):
        # FSA优化过程（与之前代码相同）
        positions = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        positions = np.round(positions)

        local_best_positions = positions.copy()
        local_best_fitness = np.array([self.evaluate(pos) for pos in positions])

        global_best_idx = np.argmin(local_best_fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = local_best_fitness[global_best_idx]

        for iter in range(self.max_iter):
            print(f"迭代 {iter + 1}/{self.max_iter}, 当前最优适应度: {global_best_fitness:.6f}")

            for i in range(self.pop_size):
                # 更新位置
                S_L = (local_best_positions[i] - positions[i]) * np.random.rand(self.dim)
                S_G = (global_best_position - positions[i]) * np.random.rand(self.dim)
                new_position = positions[i] + S_L + S_G
                new_position = np.clip(np.round(new_position), self.lb, self.ub)

                # 评估新位置
                new_fitness = self.evaluate(new_position)

                if new_fitness < local_best_fitness[i]:
                    local_best_fitness[i] = new_fitness
                    local_best_positions[i] = new_position.copy()

                    if new_fitness < global_best_fitness:
                        global_best_fitness = new_fitness
                        global_best_position = new_position.copy()

                # 更新随机初始值
                positions[i] = global_best_position + (global_best_position - local_best_positions[i]) * np.random.rand(
                    self.dim)
                positions[i] = np.clip(np.round(positions[i]), self.lb, self.ub)

        return {
            'n_estimators': int(global_best_position[0]),
            'min_samples_leaf': int(global_best_position[1]),
            'best_fitness': global_best_fitness
        }


# 主程序
def main():
    # 加载数据
    file_path = "002.csv"
    df = pd.read_csv(file_path, encoding="latin-1")

    # 数据预处理
    df = df.dropna(subset=['subscribers', 'video views', 'uploads', 'highest_yearly_earnings'])

    # 特征选择（只使用三个特征）
    features = df[['video views', 'uploads', 'highest_yearly_earnings']]
    target = df['subscribers']

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # FSA优化随机森林参数
    print("开始FSA优化...")
    optimizer = FSA_RF_Optimizer(X_train_scaled, y_train)
    best_params = optimizer.optimize()
    print("\n最优参数:", best_params)

    # 定义模型（使用优化后的随机森林参数）
    models = {
        'Random Forest (FSA优化)': RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        ),
        'Random Forest (默认)': RandomForestRegressor(random_state=42),
        'Support Vector Machine': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Naive Bayes': GaussianNB()
    }

    # 训练和评估模型
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R-squared': r2_score(y_test, y_pred),
            'Accuracy': sum(abs(y_test - y_pred) < (0.1 * y_test)) / len(y_test)
        }

    # 打印结果
    for name, metrics in results.items():
        print(f"\n{name} 模型评估结果:")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  MSE: {metrics['MSE']:.2f}")
        print(f"  R-squared: {metrics['R-squared']:.4f}")
        print(f"  准确率: {metrics['Accuracy']:.2%}")

    # 绘制对比图
    plt.figure(figsize=(15, 10))
    for i, (name, model) in enumerate(models.items(), 1):
        plt.subplot(2, 3, i)
        y_pred = model.predict(X_test_scaled)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{name}\nR² = {r2_score(y_test, y_pred):.4f}')

    plt.tight_layout()
    plt.show()

    # 特征重要性分析（仅针对随机森林模型）
    rf_model = models['Random Forest (FSA优化)']
    feature_importance = pd.DataFrame({
        '特征': features.columns,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性:")
    print(feature_importance)

    # 绘制特征重要性图
    plt.figure(figsize=(10, 5))
    plt.bar(feature_importance['特征'], feature_importance['重要性'])
    plt.title('特征重要性分析')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()