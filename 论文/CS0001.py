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

class DataVisualizer:
    """数据可视化工具类"""

    def __init__(self, df: pd.DataFrame, theme: str = 'darkgrid'):
        self.df = df
        sns.set_theme(style=theme)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False

    def set_style(self, font_scale: float = 1.5, style: str = 'darkgrid'):
        sns.set_style(style)
        sns.set(font_scale=font_scale)

    def plot_distribution(self, column: str,
                          x_label: str = None,
                          y_label: str = '频数',
                          title: str = None,
                          figsize: tuple = (10, 6),
                          kde: bool = True):
        plt.figure(figsize=figsize)
        sns.histplot(data=self.df, x=column, kde=kde)
        plt.xlabel(x_label or column)
        plt.ylabel(y_label)
        plt.title(title or f'{column}分布')
        plt.show()

    def plot_category(self, column: str,
                      x_label: str = None,
                      y_label: str = '数量',
                      plot_type: str = 'count',
                      height: float = 6,
                      aspect: float = 2,
                      title: str = None,
                      rotation: int = 45):
        g = sns.catplot(
            data=self.df,
            x=column,
            kind=plot_type,
            height=height,
            aspect=aspect
        )
        g.set_axis_labels(x_label or column, y_label)
        g.set_xticklabels(rotation=rotation)
        plt.title(title or f'{column}分布')
        plt.show()

    def plot_pie(self, column: str,
                 figsize: tuple = (8, 8),
                 title: str = None,
                 autopct: str = '%1.1f%%'):
        plt.figure(figsize=figsize)
        values = self.df[column].value_counts()
        plt.pie(values, labels=values.index, autopct=autopct)
        plt.title(title or f'{column}占比')
        plt.axis('equal')
        plt.show()

    def plot_box(self, x: str, y: str,
                 x_label: str = None,
                 y_label: str = None,
                 figsize: tuple = (10, 6),
                 title: str = None,
                 palette: str = 'GnBu_d'):
        plt.figure(figsize=figsize)
        sns.boxplot(data=self.df, x=x, y=y, palette=palette)
        plt.xlabel(x_label or x)
        plt.ylabel(y_label or y)
        plt.title(title or f'{x}与{y}的关系')
        plt.xticks(rotation=45)
        plt.show()

    def plot_top_categories(self, column: str,
                            x_label: str = None,
                            y_label: str = None,
                            n: int = 15,
                            figsize: tuple = (12, 6),
                            title: str = None,
                            horizontal: bool = True):
        data = self.df[column].value_counts()[:n]
        plt.figure(figsize=figsize)

        if horizontal:
            plt.barh(data.index, data.values)
            plt.xlabel(x_label or '数量')
            plt.ylabel(y_label or column)
        else:
            plt.bar(data.index, data.values)
            plt.xlabel(y_label or column)
            plt.ylabel(x_label or '数量')
            plt.xticks(rotation=45)

        plt.title(title or f'Top {n} {column}')
        plt.tight_layout()
        plt.show()


# 使用示例
visualizer = DataVisualizer(df)

# 1. 价格分布图
visualizer.plot_distribution(
    'log_price',
    x_label='房屋价格（对数）',
    y_label='频数',
    title='房屋价格分布'
)

# 2. 房间类型分布
visualizer.plot_category(
    'room_type',
    x_label='房间类型',
    y_label='数量',
    title='房间类型分布'
)

# 3. 房间类型占比饼图
visualizer.plot_pie(
    'room_type',
    title='房间类型占比分布'
)

# 4. 城市分布
visualizer.plot_category(
    'city',
    x_label='城市',
    y_label='数量',
    title='各城市房源分布'
)

# 5. 每个城市的价格分布
visualizer.plot_box(
    'city',
    'log_price',
    x_label='城市',
    y_label='价格（对数）',
    title='各城市房价分布'
)

# 6. 住宿类型分布
visualizer.plot_category(
    'property_type',
    x_label='住宿类型',
    y_label='数量',
    title='住宿类型分布'
)

# 7. 取消政策分布
visualizer.plot_category(
    'cancellation_policy',
    x_label='取消政策',
    y_label='数量',
    title='取消政策分布'
)

# 8. 清洁费分布
visualizer.plot_category(
    'cleaning_fee',
    x_label='清洁费',
    y_label='数量',
    title='清洁费收取情况'
)

# 9. 床型与价格关系
visualizer.plot_box(
    'bed_type',
    'log_price',
    x_label='床型',
    y_label='价格（对数）',
    title='不同床型的价格分布'
)

# 10. 床型分布
visualizer.plot_category(
    'bed_type',
    x_label='床型',
    y_label='数量',
    title='床型分布情况'
)

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
