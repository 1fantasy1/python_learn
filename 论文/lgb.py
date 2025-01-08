from CS import x_train, y_train, x_test, y_test, x
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 转换为 LightGBM 数据集格式
train_data = lgb.Dataset(x_train, label=y_train)
valid_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

# 设置模型参数
params = {
    'objective': 'regression',  # 回归任务
    'learning_rate': 0.1,       # 学习率
    'num_leaves': 31,           # 每棵树的叶子节点数
    'max_depth': -1,            # 树的最大深度
    'feature_fraction': 0.8,    # 特征选择比例
    'bagging_fraction': 0.8,    # 数据采样比例
    'bagging_freq': 5,          # 数据采样频率
    'metric': 'rmse',           # 评估指标
    'random_state': 42,         # 随机种子
    'early_stopping_rounds': 10 # 连续 10 轮没有提升则停止训练
}

# 训练模型
lgb_model = lgb.train(
    params=params,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
)

# 使用测试集预测目标值
y_pred_lgb = lgb_model.predict(x_test)

# 计算模型的性能指标
mae_lgb  = mean_absolute_error(y_test, y_pred_lgb)
mse_lgb  = mean_squared_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mse_lgb)
r2_lgb   = r2_score(y_test, y_pred_lgb)

# 输出模型的性能指标
print('\nMean Absolute Error of LightGBM Regressor     : ', mae_lgb)
print('\nMean Squared Error of LightGBM Regressor     : ', mse_lgb)
print('\nRoot Mean Squared Error of LightGBM Regressor: ', rmse_lgb)
print('\nR2 Score of LightGBM Regressor               : ', r2_lgb)

# 输出特征重要性
print("\nFeature Importances:")
importance = lgb_model.feature_importance(importance_type='gain')
for col, imp in zip(x.columns, importance):
    print(f"{col}: {imp}")

# 可视化特征重要性
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(x.columns, importance, color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()