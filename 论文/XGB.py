from 测试 import x_train, y_train, x_test, y_test, x
from xgboost import XGBRegressor  # XGBoost回归模型
from sklearn import metrics  # 性能评估工具

# 初始化 XGBoost 回归器
xgb = XGBRegressor(objective='reg:squarederror')
# - `objective='reg:squarederror'`：指定回归任务的目标函数为均方误差

# 使用训练集数据训练模型
xgb.fit(x_train, y_train)
# - `x_train`：训练特征集
# - `y_train`：训练目标变量

# 使用测试集数据进行预测
y_pred_xgb = xgb.predict(x_test)
# - `x_test`：测试特征集
# - `y_pred_xgb`：模型对测试集的预测值

# 计算模型的性能指标
mae_xgb  = metrics.mean_absolute_error(y_test, y_pred_xgb)
# 平均绝对误差 (Mean Absolute Error, MAE)：评估预测值与真实值之间的平均绝对差异

mse_xgb  = metrics.mean_squared_error(y_test, y_pred_xgb)
# 均方误差 (Mean Squared Error, MSE)：评估预测值与真实值之间的平方误差的平均值

rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
# 均方根误差 (Root Mean Squared Error, RMSE)：MSE 的平方根，表示预测误差的尺度

r2_xgb   = metrics.r2_score(y_test, y_pred_xgb)
# R² 分数 (R-squared)：衡量模型的拟合优度，1 表示完美拟合，0 表示没有预测能力

# 打印模型性能指标
print('\nMean Absolute Error of XGBoost Regressor     : ', mae_xgb)
print('\nMean Squarred Error of XGBoost Regressor     : ', mse_xgb)
print('\nRoot Mean Squarred Error of XGBoost Regressor: ', rmse_xgb)
print('\nR2 Score of XGBoost Regressor                : ', r2_xgb)