from sklearn.tree import DecisionTreeRegressor  # 决策树回归模型
from 测试 import x_train, y_train, x_test, y_test, x
from sklearn import metrics  # 性能评估工具
import numpy as np

dec_reg=DecisionTreeRegressor(criterion="squared_error",splitter="best",max_depth=5,
                              min_samples_split=4,max_features='sqrt')
dec_reg.fit(x_train, y_train)

y_pred_xgb = dec_reg.predict(x_test)

mae_dt  = metrics.mean_absolute_error(y_test, y_pred_xgb)
mse_dt  = metrics.mean_squared_error(y_test, y_pred_xgb)
rmse_dt = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
r2_dt   = metrics.r2_score(y_test, y_pred_xgb)


print('\nMean Absolute Error of Decision Tree Regressor     : ', mae_dt)
print('\nMean Squarred Error of Decision Tree Regressor     : ', mse_dt)
print('\nRoot Mean Squarred Error of Decision Tree Regressor: ', rmse_dt)
print('\nR2 Score of Decision Tree Regressor                : ', r2_dt)