from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
rcParams['font.sans-serif'] = ['Microsoft YaHei']

def ames_housing_analysis():
    print("艾姆斯房价数据回归分析:")
    housing = fetch_openml(name="house_prices", as_frame=True)  # 导入数据集
    df = housing.frame
    X = df[['GrLivArea']]  # 用地上居住面积作为自变量
    y = df['SalePrice']  # 房价作为因变量
    clf = LinearRegression()
    clf.fit(X, y)  # 模型训练
    y_pred = clf.predict(X)  # 模型预测

    # 回归方程
    print(f"回归方程:y={clf.intercept_:.1f}+{clf.coef_[0]:.1f}x")

    # 数据可视化
    plt.scatter(X, y, label='实际值', s=0.5)  # 样本实际分布
    plt.plot(X, y_pred, color='red', label='拟合曲线')  # 拟合曲线
    plt.xlabel('地上居住面积')
    plt.ylabel('房价')
    plt.title('艾姆斯房价数据回归分析')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ames_housing_analysis()