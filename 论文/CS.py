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

# 定义绘制分类图的函数
def plot_catplot(h, v, he, a):
    """
    绘制分类变量图
    参数：
    h: 横轴变量
    v: 图类型（如 'bar', 'box' 等）
    he: 图的高度
    a: 图的宽高比（aspect ratio）
    """
    sns.set(font_scale=1.5)  # 设置字体大小
    sns.catplot(x=h, kind=v, data=df, height=he, aspect=a)  # 绘制分类图
    plt.show()

# 定义绘制饼图的函数
def plot_piechart(h):
    """
    绘制饼图
    参数：
    h: 要统计的分类变量列名
    """
    sns.set(font_scale=1.5)  # 设置字体大小
    fig = plt.figure(figsize=(5, 5))  # 设置图的大小
    ax = fig.add_axes([0, 0, 1, 1])  # 添加图的坐标轴
    ax.axis('equal')  # 保证饼图为圆形
    langs = list(df[h].unique())  # 获取分类变量的唯一值列表
    students = list(df[h].value_counts())  # 获取每个分类的频率
    ax.pie(students, labels=langs, autopct='%1.2f%%')  # 绘制饼图并设置标签和百分比格式
    plt.show()  # 显示饼图

# 定义绘制箱线图的函数
def plot_boxplot(h, v):
    """
    绘制箱线图
    参数：
    h: 横轴变量（分类变量）
    v: 纵轴变量（连续变量）
    """
    plt.figure(figsize=(10, 8))  # 设置图形大小
    sns.set(font_scale=1.5)  # 设置字体大小
    sns.boxplot(data=df, x=h, y=v, palette='GnBu_d')  # 绘制箱线图
    plt.title('Density and distribution of prices ', fontsize=15)  # 设置标题
    plt.xlabel(h)  # 设置横轴标签
    plt.ylabel(v)  # 设置纵轴标签
    plt.show()

# 绘制价格分布直方图
# plt.figure(figsize=(8, 6))  # 设置图的大小
# sns.distplot(df["log_price"])  # 使用Seaborn绘制分布图
# plt.title('Price distribution')  # 设置标题
# plt.show()
# 1.png

# 绘制分类变量 "room_type" 的计数图
# plot_catplot("room_type", "count", 5, 2)
# 2.png

# 绘制 "room_type" 的饼图，展示不同房间类型的比例分布
# plot_piechart("room_type")
# 3.png

# 绘制分类变量 "city" 的计数图，展示不同城市的房源数量
# plot_catplot("city", "count", 5, 2)
# 4.png

# 绘制城市分布的饼图，展示每个城市房源的比例
# fig = plt.figure(figsize=(5, 5))  # 设置图的大小为 5x5
# ax = fig.add_axes([0, 0, 1, 1])  # 添加坐标轴
# ax.axis('equal')  # 确保饼图为圆形
# langs = list(df.city.unique())
# students = list(df.city.value_counts())
# ax.pie(students, labels=langs, autopct='%1.2f%%')
# 5.png

# 获取 "neighbourhood" 列中出现频率最高的前15个社区
# data = df.neighbourhood.value_counts()[:15]
# plt.figure(figsize=(22, 22))
# x = list(data.index)  # 社区名称
# y = list(data.values)  # 每个社区的房源数量
# x.reverse()
# y.reverse()
# plt.title("Most popular Neighbourhood")  # 图标题
# plt.ylabel("Neighbourhood Area")  # y轴标签
# plt.xlabel("Number of guest who host in this area")  # x轴标签
# plt.barh(x, y)  # `barh` 绘制水平条形图
# plt.show()
# 6.png

# 使用 plot_catplot 绘制 "cancellation_policy" 的分类图
# plot_catplot("cancellation_policy", "count", 10, 2)
# 7.png

# 使用 plot_catplot 绘制 "cleaning_fee" 的分类图
# plot_catplot("cleaning_fee", "count", 6, 2)
# 8.png

# 绘制 "city" 与 "log_price" 的箱线图
# plot_boxplot("city", "log_price")
# 9.png

# 绘制 "room_type" 与 "log_price" 的箱线图
# plot_boxplot("room_type", "log_price")
# 10.png

# 绘制 "cancellation_policy" 与 "log_price" 的箱线图
# plot_boxplot("cancellation_policy", "log_price")
# 11.png

# 绘制 "bed_type" 与 "log_price" 的箱线图
# plot_boxplot("bed_type", "log_price")
# 12.png

# 使用 plot_catplot 绘制 "bed_type" 的分类计数图
# plot_catplot("bed_type", "count", 8, 2)
# 13.png

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

# print(new_df)

# 划分特征和目标变量
x = new_df.drop('log_price', axis=1)  # 特征集（去掉目标列 'log_price'）
y = new_df['log_price']  # 目标变量（价格列）

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 对特征进行标准化处理
sc = StandardScaler()  # 初始化标准化工具
x_train = sc.fit_transform(x_train)  # 对训练集进行标准化（均值为0，方差为1）
x_test = sc.transform(x_test)  # 使用同样的参数对测试集进行标准化

lr = LinearRegression()

lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)

mae_lr = metrics.mean_absolute_error(y_test, y_pred_lr)
mse_lr = metrics.mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
r2_lr = metrics.r2_score(y_test, y_pred_lr)


print('\nMean Absolute Error of Linear Regression     : ', mae_lr)
print('\nMean Squarred Error of Linear Regression     : ', mse_lr)
print('\nRoot Mean Squarred Error of Linear Regression: ', rmse_lr)
print('\nR2 Score of Linear Regression                : ', r2_lr)

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
·

model_CBR = CatBoostRegressor(verbose=100)

model_CBR.fit(x_train, y_train)

cross_val_score(model_CBR, x_train, y_train,
                           scoring='r2',
                           cv=KFold(n_splits=5,
                                    shuffle=True,
                                    random_state=42,
                                    ))

y_pred_cbr = model_CBR.predict(x_test)

mae_cbr  = metrics.mean_absolute_error(y_test, y_pred_cbr)
mse_cbr  = metrics.mean_squared_error(y_test, y_pred_cbr)
rmse_cbr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_cbr))
r2_cbr   = metrics.r2_score(y_test, y_pred_cbr)

print('\nMean Absolute Error of CatBoost Regressor     : ', mae_cbr)
print('\nMean Squarred Error of CatBoost Regressor     : ', mse_cbr)
print('\nRoot Mean Squarred Error of CatBoost Regressor: ', rmse_cbr)
print('\nR2 Score of CatBoost Regressor                : ', r2_cbr)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

gb.fit(x_train, y_train)

y_pred_gb = gb.predict(x_test)

mae_gb  = metrics.mean_absolute_error(y_test, y_pred_gb)
mse_gb  = metrics.mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gb))
r2_gb   = metrics.r2_score(y_test, y_pred_gb)


print('\nMean Absolute Error of Gradient Boosting     : ', mae_gb)
print('\nMean Squarred Error of Gradient Boosting     : ', mse_gb)
print('\nRoot Mean Squarred Error of Gradient Boosting: ', rmse_gb)
print('\nR2 Score of Gradient Boosting                : ', r2_gb)

xgb = XGBRegressor(objective='reg:squarederror')
xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)

mae_xgb  = metrics.mean_absolute_error(y_test, y_pred_xgb)
mse_xgb  = metrics.mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
r2_xgb   = metrics.r2_score(y_test, y_pred_xgb)


print('\nMean Absolute Error of XGBoost Regressor     : ', mae_xgb)
print('\nMean Squarred Error of XGBoost Regressor     : ', mse_xgb)
print('\nRoot Mean Squarred Error of XGBoost Regressor: ', rmse_xgb)
print('\nR2 Score of XGBoost Regressor                : ', r2_xgb)

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
