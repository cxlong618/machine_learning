"""
线性回归完整示例
包含：数据生成、模型训练、评估、可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from utils import config, ensure_chinese_font

# 应用字体修复
ensure_chinese_font()
print("✅ 已加载配置，中文字体已设置")

# 1. 生成示例数据
def generate_sample_data(n_samples=100, noise=0.1):
    """
    生成一元线性回归的示例数据
    y = 3x + 2 + noise
    """
    np.random.seed(42)
    X = np.random.uniform(0, 10, n_samples).reshape(-1, 1)
    y = 3 * X.flatten() + 2 + np.random.normal(0, noise, n_samples)
    return X, y

# 2. 创建模型类
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, X, y, test_size=0.2):
        """数据预处理和分割"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"训练集大小: {self.X_train.shape[0]}")
        print(f"测试集大小: {self.X_test.shape[0]}")

    def train(self):
        """训练模型"""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """预测"""
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        return y_train_pred, y_test_pred

    def evaluate(self):
        """评估模型"""
        y_train_pred, y_test_pred = self.predict()

        # 训练集评估
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)

        # 测试集评估
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)

        print("\n=== 模型评估 ===")
        print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

        return {
            'train_mse': train_mse, 'train_r2': train_r2,
            'test_mse': test_mse, 'test_r2': test_r2
        }

    def get_coefficients(self):
        """获取模型参数"""
        coef = self.model.coef_[0]
        intercept = self.model.intercept_
        print(f"\n=== 模型参数 ===")
        print(f"斜率 (权重): {coef:.4f}")
        print(f"截距 (偏置): {intercept:.4f}")
        return coef, intercept

    def visualize_results(self):
        """可视化结果"""
        y_train_pred, y_test_pred = self.predict()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 子图1：训练集散点图和回归线
        ax1.scatter(self.X_train, self.y_train, alpha=0.6, label='真实值')
        ax1.plot(self.X_train, y_train_pred, 'r-', label='预测值')
        ax1.set_title('训练集拟合结果')
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：测试集散点图和回归线
        ax2.scatter(self.X_test, self.y_test, alpha=0.6, label='真实值')
        ax2.plot(self.X_test, y_test_pred, 'r-', label='预测值')
        ax2.set_title('测试集预测结果')
        ax2.set_xlabel('X')
        ax2.set_ylabel('y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3：残差图
        residuals_train = self.y_train - y_train_pred
        residuals_test = self.y_test - y_test_pred

        ax3.scatter(y_train_pred, residuals_train, alpha=0.6, label='训练集')
        ax3.scatter(y_test_pred, residuals_test, alpha=0.6, label='测试集')
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('残差图')
        ax3.set_xlabel('预测值')
        ax3.set_ylabel('残差')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4：真实值vs预测值
        ax4.scatter(self.y_train, y_train_pred, alpha=0.6, label='训练集')
        ax4.scatter(self.y_test, y_test_pred, alpha=0.6, label='测试集')

        # 添加对角线
        min_val = min(self.y_train.min(), self.y_test.min(), y_train_pred.min(), y_test_pred.min())
        max_val = max(self.y_train.max(), self.y_test.max(), y_train_pred.max(), y_test_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')

        ax4.set_title('真实值 vs 预测值')
        ax4.set_xlabel('真实值')
        ax4.set_ylabel('预测值')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config.get_plot_path('linear_regression_results.png'), dpi=300, bbox_inches='tight')
        plt.show()

# 3. 多元线性回归示例
def multiple_linear_regression_example():
    """多元线性回归示例：房价预测"""
    print("\n" + "="*50)
    print("多元线性回归示例：房价预测")
    print("="*50)

    # 创建模拟房价数据
    np.random.seed(42)
    n_samples = 200

    # 特征：房间数量、面积、房龄
    rooms = np.random.randint(1, 6, n_samples)
    area = np.random.uniform(50, 200, n_samples)
    age = np.random.uniform(0, 30, n_samples)

    # 价格 = 基础价 + 房间系数*房间数 + 面积系数*面积 + 房龄系数*房龄 + 噪声
    price = (100 + 20 * rooms + 0.5 * area - 2 * age +
             np.random.normal(0, 10, n_samples))

    # 创建DataFrame
    df = pd.DataFrame({
        'rooms': rooms,
        'area': area,
        'age': age,
        'price': price
    })

    print("数据预览:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")

    # 准备数据
    X = df[['rooms', 'area', 'age']]
    y = df['price']

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测和评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n模型评估:")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 显示系数
    print(f"\n模型系数:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"截距: {model.intercept_:.4f}")

    return model

# 4. 正则化线性回归
def regularization_example():
    """正则化线性回归示例"""
    print("\n" + "="*50)
    print("正则化线性回归示例")
    print("="*50)

    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler

    # 生成具有多重共线性的数据
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 10)

    # 创建多重共线性特征
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] * 2 + 0.1 * np.random.randn(n_samples)

    # 真实系数（大部分为0，模拟稀疏性）
    true_coef = np.array([1.5, 0, 0, -1.0, 0, 0, 0.8, 0, 0, -0.5])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 比较不同正则化方法
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1.0)': Lasso(alpha=1.0)
    }

    print("不同正则化方法的比较:")
    print("-" * 70)
    print(f"{'模型':<20} {'MSE':<10} {'R²':<10} {'非零系数数量':<10}")
    print("-" * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if hasattr(model, 'coef_'):
            non_zero_coefs = np.sum(model.coef_ != 0)
        else:
            non_zero_coefs = '-'

        print(f"{name:<20} {mse:<10.4f} {r2:<10.4f} {non_zero_coefs:<10}")

# 主函数
def main():
    """主函数：运行所有示例"""
    print("="*60)
    print("线性回归完整教程")
    print("="*60)

    # 1. 生成数据
    X, y = generate_sample_data(n_samples=100, noise=0.5)

    # 2. 创建并训练简单线性回归模型
    model = LinearRegressionModel()
    model.prepare_data(X, y)
    model.train()

    # 3. 评估模型
    metrics = model.evaluate()

    # 4. 显示模型参数
    coef, intercept = model.get_coefficients()

    # 5. 可视化结果
    model.visualize_results()

    # 6. 多元线性回归示例
    multiple_linear_regression_example()

    # 7. 正则化示例
    regularization_example()

    print("\n" + "="*60)
    print("线性回归教程完成！")
    print("="*60)

if __name__ == "__main__":
    main()