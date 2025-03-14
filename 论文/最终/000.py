import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 加载数据集
data = pd.read_csv('Airbnb_Data.csv')
df = data.copy()

# 输出数据的形状（行数和列数）
# print(df.shape)

# 数据的基本信息（如数据类型和是否有缺失值）
# print(df.info())

# 删除不需要的列，简化数据集
new_df = df.drop(
    [
        'id', 'name', 'description', 'first_review', 'host_since',
        'host_has_profile_pic', 'host_identity_verified', 'last_review',
        'neighbourhood', 'thumbnail_url', 'zipcode', 'host_response_rate'
    ],
    axis=1
)

# 填充缺失值
new_df["bathrooms"] = df['bathrooms'].fillna(round(df["bathrooms"].median()))
new_df["review_scores_rating"] = df["review_scores_rating"].fillna(60)
new_df["bedrooms"] = df['bedrooms'].fillna((df["bathrooms"].median()))
new_df["beds"] = df["beds"].fillna((df["bathrooms"].median()))

'''
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
    sns.catplot(x=h, kind=v, data=new_df, height=he, aspect=a)  # 绘制分类图
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
    langs = list(new_df[h].unique())  # 获取分类变量的唯一值列表
    students = list(new_df[h].value_counts())  # 获取每个分类的频率
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
    sns.boxplot(data=new_df, x=h, y=v, palette='GnBu_d')  # 绘制箱线图
    plt.title('Density and distribution of prices ', fontsize=15)  # 设置标题
    plt.xlabel(h)  # 设置横轴标签
    plt.ylabel(v)  # 设置纵轴标签
    plt.show()

# 绘制价格分布直方图
plt.figure(figsize=(8, 6))  # 设置图的大小
sns.distplot(new_df["log_price"])  # 使用Seaborn绘制分布图
plt.title('Price distribution')  # 设置标题
plt.show()
# 1.png

# 绘制分类变量 "room_type" 的计数图
plot_catplot("room_type", "count", 5, 2)
# 2.png

# 绘制 "room_type" 的饼图，展示不同房间类型的比例分布
plot_piechart("room_type")
# 3.png

# 绘制分类变量 "city" 的计数图，展示不同城市的房源数量
plot_catplot("city", "count", 5, 2)
# 4.png

# 绘制城市分布的饼图，展示每个城市房源的比例
fig = plt.figure(figsize=(5, 5))  # 设置图的大小为 5x5
ax = fig.add_axes([0, 0, 1, 1])  # 添加坐标轴
ax.axis('equal')  # 确保饼图为圆形
langs = list(new_df.city.unique())
students = list(new_df.city.value_counts())
ax.pie(students, labels=langs, autopct='%1.2f%%')
# 5.png

# 获取 "neighbourhood" 列中出现频率最高的前15个社区
data = df.neighbourhood.value_counts()[:15]
plt.figure(figsize=(22, 22))
x = list(data.index)  # 社区名称
y = list(data.values)  # 每个社区的房源数量
x.reverse()
y.reverse()
plt.title("Most popular Neighbourhood")  # 图标题
plt.ylabel("Neighbourhood Area")  # y轴标签
plt.xlabel("Number of guest who host in this area")  # x轴标签
plt.barh(x, y)  # `barh` 绘制水平条形图
plt.show()
# 6.png

# 使用 plot_catplot 绘制 "cancellation_policy" 的分类图
plot_catplot("cancellation_policy", "count", 10, 2)
# 7.png

# 使用 plot_catplot 绘制 "cleaning_fee" 的分类图
plot_catplot("cleaning_fee", "count", 6, 2)
# 8.png

# 绘制 "city" 与 "log_price" 的箱线图
plot_boxplot("city", "log_price")
# 9.png

# 绘制 "room_type" 与 "log_price" 的箱线图
plot_boxplot("room_type", "log_price")
# 10.png

# 绘制 "cancellation_policy" 与 "log_price" 的箱线图
plot_boxplot("cancellation_policy", "log_price")
# 11.png

# 绘制 "bed_type" 与 "log_price" 的箱线图
plot_boxplot("bed_type", "log_price")
# 12.png

# 使用 plot_catplot 绘制 "bed_type" 的分类计数图
plot_catplot("bed_type", "count", 8, 2)
# 13.png
'''

categorical_col = []  # 用于存储分类变量的列名
numerical_col = []  # 用于存储数值变量的列名

# 遍历数据集的所有列，根据数据类型将列分类
for column in new_df.columns:
    if new_df[column].dtypes != "float64" and new_df[column].dtypes != "int64":
        categorical_col.append(column)
    else:
        numerical_col.append(column)
le = LabelEncoder()
for col in categorical_col:
    new_df[col] = le.fit_transform(new_df[col])
pd.set_option("display.max_columns", None)

# 划分特征和目标变量
x = new_df.drop('log_price', axis=1)
y = new_df['log_price']

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 对特征进行标准化处理
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

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
print("\n=========== 所有模型的最终结果 ===========")
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