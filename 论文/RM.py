from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归模型
from 测试 import x_train, y_train, x_test, y_test, x
from sklearn import metrics  # 性能评估工具
import numpy as np


ridge_model=Ridge()

ridge_model.fit(x_train,y_train)

y_pred_lr = ridge_model.predict(x_test)

mae_rd = metrics.mean_absolute_error(y_test, y_pred_lr)
mse_rd = metrics.mean_squared_error(y_test, y_pred_lr)
rmse_rd = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
r2_rd = metrics.r2_score(y_test, y_pred_lr)


print('\nMean Absolute Error of Ridge Regression     : ', mae_rd)
print('\nMean Squarred Error of Ridge Regression     : ', mse_rd)
print('\nRoot Mean Squarred Error of Ridge Regression: ', rmse_rd)
print('\nR2 Score of Ridge Regression                : ', r2_rd)