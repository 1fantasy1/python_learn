from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # 集成方法
from 测试 import x_train, y_train, x_test, y_test, x
from sklearn import metrics  # 性能评估工具
import numpy as np

rmf_reg=RandomForestRegressor(n_estimators=100,criterion="squared_error",
                              bootstrap=True,oob_score=True)
rmf_reg.fit(x_train, y_train)

y_pred_xgb = rmf_reg.predict(x_test)

mae_rf  = metrics.mean_absolute_error(y_test, y_pred_xgb)
mse_rf  = metrics.mean_squared_error(y_test, y_pred_xgb)
rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
r2_rf  = metrics.r2_score(y_test, y_pred_xgb)


print('\nMean Absolute Error of Random Forest Regressor     : ', mae_rf)
print('\nMean Squarred Error of Random Forest Regressor     : ', mse_rf)
print('\nRoot Mean Squarred Error of Random Forest Regressor: ', rmse_rf)
print('\nR2 Score of Random Forest Regressor                : ', r2_rf)