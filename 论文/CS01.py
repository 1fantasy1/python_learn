import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# 加载数据集
data = pd.read_csv('Airbnb_Data.csv')  # 从CSV文件中读取数据
df = data.copy()  # 创建数据副本，避免直接修改原始数据

# 删除不需要的列，简化数据集
new_df = df.drop(
    [
        'id', 'name', 'description', 'first_review', 'host_since',
        'host_has_profile_pic', 'host_identity_verified', 'last_review',
        'neighbourhood', 'thumbnail_url', 'zipcode', 'host_response_rate'
    ],
    axis=1  # 指定按列删除
)

# 填充缺失值
new_df["bathrooms"] = df['bathrooms'].fillna(round(df["bathrooms"].median()))
new_df["review_scores_rating"] = df["review_scores_rating"].fillna(0)
new_df["bedrooms"] = df['bedrooms'].fillna((df["bathrooms"].median()))
new_df["beds"] = df["beds"].fillna((df["bathrooms"].median()))

categorical_col = []  # 用于存储分类变量的列名
numerical_col = []  # 用于存储数值变量的列名

# 遍历数据集的所有列，根据数据类型将列分类
for column in new_df.columns:
    if new_df[column].dtypes != "float64" and new_df[column].dtypes != "int64":
        categorical_col.append(column)  # 如果不是数值类型，添加到分类变量列表
    else:
        numerical_col.append(column)  # 如果是数值类型，添加到数值变量列表
le = LabelEncoder()
for col in categorical_col:
    new_df[col] = le.fit_transform(new_df[col])  # 将分类变量转换为数值类型
pd.set_option("display.max_columns", None)

# 划分特征和目标变量
x = new_df.drop('log_price', axis=1)  # 特征集（去掉目标列 'log_price'）
y = new_df['log_price']  # 目标变量（价格列）

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 对特征进行标准化处理
sc = StandardScaler()  # 初始化标准化工具
x_train = sc.fit_transform(x_train)  # 对训练集进行标准化（均值为0，方差为1）
x_test = sc.transform(x_test)  # 使用同样的参数对测试集进行标准化

# # 定义性能评估函数
# def evaluate_model(y_true, y_pred, model_name):
#     mae = metrics.mean_absolute_error(y_true, y_pred)
#     mse = metrics.mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = metrics.r2_score(y_true, y_pred)
#     print(f'\n{model_name} 性能表现:')  # 输出模型名称及其性能表现
#     print(f'平均绝对误差      : {mae}')  # 输出平均绝对误差
#     print(f'均方误差         : {mse}')  # 输出均方误差
#     print(f'均方根误差       : {rmse}')  # 输出均方根误差
#     print(f'R2得分          : {r2}')  # 输出 R2 得分
#     return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
#
# # 模型初始化
# models = {
#     '线性回归': LinearRegression(),
#     '决策树': DecisionTreeRegressor(criterion="squared_error", splitter="best", max_depth=5,
#                                            min_samples_split=4, max_features='sqrt', random_state=42),
#     '随机森林': RandomForestRegressor(n_estimators=100, criterion="squared_error",
#                                            bootstrap=True, oob_score=True, random_state=42),
#     'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
# }
#
# # 模型训练和评估
# results = []
# for model_name, model in models.items():
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     result = evaluate_model(y_test, y_pred, model_name)
#     results.append(result)
#
# # 将结果输出为表格
# results_df = pd.DataFrame(results)
# print("\n模型性能摘要:")
# print(results_df)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np


def evaluate_model(y_true, y_pred, model_name):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f'\n{model_name} 性能表现:')
    print(f'平均绝对误差      : {mae}')
    print(f'均方误差         : {mse}')
    print(f'均方根误差       : {rmse}')
    print(f'R2得分          : {r2}')
    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


# 精简后的参数网格
param_grids = {
    '线性回归': {
        'fit_intercept': [True, False]
    },

    '决策树': {
        'max_depth': [5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },

    '随机森林': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },

    'XGBoost': {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

# 模型初始化
models = {
    '线性回归': LinearRegression(),
    '决策树': DecisionTreeRegressor(random_state=42),
    '随机森林': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
}

# 模型训练和评估
results = []
for model_name, model in models.items():
    print(f'\n开始训练 {model_name} ...')

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=5,  # 5折交叉验证
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # 训练模型
    grid_search.fit(x_train, y_train)

    # 输出最佳参数
    print(f'\n{model_name} 最佳参数:')
    print(grid_search.best_params_)
    print(f'最佳交叉验证得分: {-grid_search.best_score_:.4f}')

    # 使用最佳模型进行预测
    y_pred = grid_search.predict(x_test)

    # 评估模型性能
    result = evaluate_model(y_test, y_pred, model_name)
    result['Best_Parameters'] = grid_search.best_params_
    result['Best_CV_Score'] = -grid_search.best_score_
    results.append(result)

# 打印所有模型的最终结果
print("\n=== 所有模型的最终结果 ===")
for result in results:
    print(f"\n{'-' * 40}")
    print(f"模型: {result['Model']}")
    print(f"最佳交叉验证得分: {result['Best_CV_Score']:.4f}")
    print("\n最佳参数:")
    for param, value in result['Best_Parameters'].items():
        print(f"{param}: {value}")
    print("\n测试集性能:")
    print(f"MAE: {result['MAE']:.4f}")
    print(f"MSE: {result['MSE']:.4f}")
    print(f"RMSE: {result['RMSE']:.4f}")
    print(f"R2: {result['R2']:.4f}")