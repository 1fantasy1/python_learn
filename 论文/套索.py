from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归模型
from 测试 import x_train, y_train, x_test, y_test, x
from sklearn import metrics  # 性能评估工具

lasso=Lasso()

lasso.fit(x_train,y_train)

y_pred_lr = lasso.predict(x_test)

mae_lso = metrics.mean_absolute_error(y_test, y_pred_lr)
mse_lso = metrics.mean_squared_error(y_test, y_pred_lr)
rmse_lso = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
r2_lso = metrics.r2_score(y_test, y_pred_lr)


print('\nMean Absolute Error of Lasso Regression     : ', mae_lso)
print('\nMean Squarred Error of Lasso Regression     : ', mse_lso)
print('\nRoot Mean Squarred Error of Lasso Regression: ', rmse_lso)
print('\nR2 Score of Lasso Regression                : ', r2_lso)