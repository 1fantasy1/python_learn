import pandas as pd
import numpy as np
import cupy as cp  # 导入cupy用于GPU加速
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class FSA_RF_Optimizer:
    def __init__(self, X, y, pop_size=30, max_iter=50):
        print("使用GPU加速计算...")

        # 确保输入数据是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 数据集分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 转换为CuPy数组进行GPU加速
        self.X_train = cp.array(self.X_train)
        self.X_test = cp.array(self.X_test)
        self.y_train = cp.array(self.y_train)
        self.y_test = cp.array(self.y_test)

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = 2
        self.lb = cp.array([10, 1])
        self.ub = cp.array([200, 20])
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

        # 转回numpy进行sklearn评估
        X_train_np = cp.asnumpy(self.X_train)
        y_train_np = cp.asnumpy(self.y_train)

        scores = cross_val_score(rf, X_train_np, y_train_np,
                                 cv=5, scoring='neg_mean_squared_error')
        return -float(cp.mean(cp.array(scores)))

    def optimize(self):
        # 使用CuPy生成随机数
        positions = self.lb + (self.ub - self.lb) * cp.random.rand(self.pop_size, self.dim)
        positions = cp.round(positions)

        local_best_positions = positions.copy()
        local_best_fitness = cp.array([self.evaluate(pos) for pos in positions])

        global_best_idx = int(cp.argmin(local_best_fitness))
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = float(local_best_fitness[global_best_idx])

        for iter in range(self.max_iter):
            print(f"迭代 {iter + 1}/{self.max_iter}, 当前最优适应度: {global_best_fitness:.6f}")

            for i in range(self.pop_size):
                # 使用CuPy进行计算
                S_L = (local_best_positions[i] - positions[i]) * cp.random.rand(self.dim)
                S_G = (global_best_position - positions[i]) * cp.random.rand(self.dim)
                new_position = positions[i] + S_L + S_G
                new_position = cp.clip(cp.round(new_position), self.lb, self.ub)

                new_fitness = self.evaluate(new_position)

                if new_fitness < local_best_fitness[i]:
                    local_best_fitness[i] = new_fitness
                    local_best_positions[i] = new_position.copy()

                    if new_fitness < global_best_fitness:
                        global_best_fitness = new_fitness
                        global_best_position = new_position.copy()

                positions[i] = global_best_position + (global_best_position - local_best_positions[i]) * cp.random.rand(
                    self.dim)
                positions[i] = cp.clip(cp.round(positions[i]), self.lb, self.ub)

        # 转回numpy数组返回结果
        return {
            'n_estimators': int(cp.asnumpy(global_best_position)[0]),
            'min_samples_leaf': int(cp.asnumpy(global_best_position)[1]),
            'best_fitness': float(global_best_fitness)
        }


def main():
    # 加载数据
    file_path = "002.csv"
    df = pd.read_csv(file_path, encoding="latin-1")

    # 数据预处理
    df = df.dropna(subset=['subscribers', 'video views', 'uploads', 'highest_yearly_earnings'])

    # 特征选择（三个特征）
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

    # 定义模型
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

    # 训练和评估
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

    # 特征重要性分析
    rf_model = models['Random Forest (FSA优化)']
    feature_importance = pd.DataFrame({
        '特征': features.columns,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性:")
    print(feature_importance)

    plt.figure(figsize=(10, 5))
    plt.bar(feature_importance['特征'], feature_importance['重要性'])
    plt.title('特征重要性分析')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 清理GPU内存
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


if __name__ == "__main__":
    main()