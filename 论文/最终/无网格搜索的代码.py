from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from 有网格搜索的代码 import x_train, y_train, x_test, y_test


def evaluate_model(y_true, y_pred, model_name):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)
    accuracy_5 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.05) * 100
    accuracy_10 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100

    print(f'\n{model_name} 性能表现:')
    print(f'平均绝对误差      : {mae:.4f}')
    print(f'均方误差         : {mse:.4f}')
    print(f'均方根误差       : {rmse:.4f}')
    print(f'R2得分          : {r2:.4f}')
    print(f'5%误差范围内准确率: {accuracy_5:.2f}%')
    print(f'10%误差范围内准确率: {accuracy_10:.2f}%')

    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy_5': accuracy_5,
        'Accuracy_10': accuracy_10
    }


# 模型初始化
models = {
    '线性回归': LinearRegression(),
    '决策树': DecisionTreeRegressor(max_depth=10, random_state=42),
    '随机森林': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    '支持向量机': SVR(kernel='linear', C=0.1, epsilon=0.1)
}

# 模型训练和评估
results = []

for model_name, model in models.items():
    print(f'\n{"=" * 50}')
    print(f'开始训练 {model_name} ...')

    # 训练模型
    model.fit(x_train, y_train)

    # 预测
    y_pred = model.predict(x_test)

    # 评估模型性能
    result = evaluate_model(y_test, y_pred, model_name)
    results.append(result)

# 绘制所有模型的准确率对比
plt.figure(figsize=(12, 6))
model_names = [r['Model'] for r in results]
acc_5 = [r['Accuracy_5'] for r in results]
acc_10 = [r['Accuracy_10'] for r in results]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width / 2, acc_5, width, label='5%误差范围内准确率')
plt.bar(x + width / 2, acc_10, width, label='10%误差范围内准确率')

plt.ylabel('准确率 (%)')
plt.title('模型准确率对比')
plt.xticks(x, model_names)
plt.legend()

# 在柱状图上添加数值标签
for i, acc in enumerate(acc_5):
    plt.text(i - width / 2, acc, f'{acc:.1f}%', ha='center', va='bottom')
for i, acc in enumerate(acc_10):
    plt.text(i + width / 2, acc, f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 打印所有模型的最终结果
print("\n=== 所有模型的最终结果 ===")
for result in results:
    print(f"\n{'-' * 50}")
    print(f"模型: {result['Model']}")
    print("\n测试集性能:")
    print(f"MAE: {result['MAE']:.4f}")
    print(f"MSE: {result['MSE']:.4f}")
    print(f"RMSE: {result['RMSE']:.4f}")
    print(f"R2: {result['R2']:.4f}")
    print(f"5%误差范围内准确率: {result['Accuracy_5']:.2f}%")
    print(f"10%误差范围内准确率: {result['Accuracy_10']:.2f}%")