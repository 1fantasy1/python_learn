
# 转换为 LightGBM 数据集格式
train_data = lgb.Dataset(x_train, label=y_train)
valid_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

# 设置模型参数
params = {
    'objective': 'regression',  # 回归任务
    'learning_rate': 0.1,       # 学习率
    'verbose': -1,
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

r2_list = {"Linear Regression": r2_lr,
           "Ridge Regression":r2_rd,
           "Lasso Regression": r2_lso,
          "CatBoost": r2_cbr,
          "Gradient Boosting":r2_gb ,
          "XGBoost": r2_xgb,
           "Decision Tree": r2_dt,
          "Random Forest": r2_rf,
           "LightGBM": r2_lgb}

mae_list = {"Linear Regression": mae_lr,
           "Ridge Regression": mae_rd,
           "Lasso Regression": mae_lso,
          "CatBoost": mae_cbr,
          "Gradient Boosting": mae_gb ,
          "XGBoost": mae_xgb,
           "Decision Tree": mae_dt,
          "Random Forest": mae_rf,
            "LightGBM": mae_lgb}

mse_list = {"Linear Regression": mse_lr,
           "Ridge Regression": mse_rd,
           "Lasso Regression": mse_lso,
          "CatBoost": mse_cbr,
          "Gradient Boosting": mse_gb ,
          "XGBoost": mse_xgb,
           "Decision Tree": mse_dt,
          "Random Forest": mse_rf,
            "LightGBM": mse_lgb}

rmse_list = {"Linear Regression": rmse_lr,
           "Ridge Regression": rmse_rd,
           "Lasso Regression": rmse_lso,
          "CatBoost": rmse_cbr,
          "Gradient Boosting": rmse_gb ,
          "XGBoost": rmse_xgb,
           "Decision Tree": rmse_dt,
          "Random Forest": rmse_rf,
             "LightGBM": rmse_lgb}

a1 =  pd.DataFrame.from_dict(r2_list, orient = 'index', columns = ["R2 SCORE"])
a2 =  pd.DataFrame.from_dict(mae_list, orient = 'index', columns = ["MEAN ABSOLUTE ERROR"])
a3 =  pd.DataFrame.from_dict(mse_list, orient = 'index', columns = ["MEAN SQUARRED ERROR"])
a4 =  pd.DataFrame.from_dict(rmse_list, orient = 'index', columns = ["ROOT MEAN SQUARRED ERROR"])

org = pd.concat([a1, a2, a3, a4], axis = 1)
print(org)

import matplotlib.pyplot as plt

org["R2 SCORE"].plot(kind="bar", figsize=(10, 6), color="skyblue", edgecolor="black")
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
