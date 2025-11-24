"""
分类算法对比示例
包含：逻辑回归、决策树、随机森林、SVM、KNN
使用鸢尾花数据集进行演示
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from utils import config, ensure_chinese_font

# 应用字体修复
ensure_chinese_font()
print("✅ 已加载配置，中文字体已设置")

class ClassificationComparison:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}

    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("="*50)
        print("1. 数据加载和预处理")
        print("="*50)

        # 加载鸢尾花数据集
        iris = load_iris()
        self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.data['target'] = iris.target
        self.data['target_name'] = [iris.target_names[i] for i in iris.target]

        print("数据集信息:")
        print(f"样本数量: {len(self.data)}")
        print(f"特征数量: {len(iris.feature_names)}")
        print(f"类别数量: {len(iris.target_names)}")
        print(f"类别名称: {iris.target_names}")

        print("\n数据预览:")
        print(self.data.head())

        print("\n各类别样本数量:")
        print(self.data['target_name'].value_counts())

        # 分离特征和标签
        X = iris.data
        y = iris.target

        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\n训练集大小: {self.X_train.shape[0]}")
        print(f"测试集大小: {self.X_test.shape[0]}")

    def initialize_models(self):
        """初始化各种分类模型"""
        print("\n" + "="*50)
        print("2. 模型初始化")
        print("="*50)

        self.models = {
            '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
            '决策树': DecisionTreeClassifier(random_state=42),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        print("已初始化的模型:")
        for name in self.models.keys():
            print(f"- {name}")

    def train_and_evaluate_models(self):
        """训练和评估所有模型"""
        print("\n" + "="*50)
        print("3. 模型训练和评估")
        print("="*50)

        for name, model in self.models.items():
            print(f"\n训练 {name}...")

            # 训练模型
            if name in ['逻辑回归', 'SVM (RBF)', 'KNN']:
                # 使用标准化数据
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
            else:
                # 不需要标准化的模型
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

            # 计算评估指标
            accuracy = accuracy_score(self.y_test, y_pred)

            # 交叉验证
            if name in ['逻辑回归', 'SVM (RBF)', 'KNN']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            # 保存结果
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"测试集准确率: {accuracy:.4f}")
            print(f"交叉验证平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    def visualize_results(self):
        """可视化结果"""
        print("\n" + "="*50)
        print("4. 结果可视化")
        print("="*50)

        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        # 1. 准确率对比柱状图
        ax = axes[0]
        names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in names]
        cv_means = [self.results[name]['cv_mean'] for name in names]
        cv_stds = [self.results[name]['cv_std'] for name in names]

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width/2, accuracies, width, label='测试集准确率', alpha=0.8)
        ax.bar(x + width/2, cv_means, width, yerr=cv_stds, label='交叉验证准确率', alpha=0.8)

        ax.set_xlabel('模型')
        ax.set_ylabel('准确率')
        ax.set_title('模型准确率对比')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2-6. 混淆矩阵
        target_names = load_iris().target_names
        for i, (name, result) in enumerate(self.results.items(), 1):
            if i >= 6:  # 最多显示5个模型
                break

            ax = axes[i]
            cm = confusion_matrix(self.y_test, result['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=target_names, yticklabels=target_names)
            ax.set_title(f'{name} - 混淆矩阵')
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')

        plt.tight_layout()
        plt.savefig(config.get_plot_path('classification_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 特征重要性（针对树模型）
        self.plot_feature_importance()

        # 4. 决策树可视化
        if '决策树' in self.results:
            self.plot_decision_tree()

    def plot_feature_importance(self):
        """绘制特征重要性"""
        print("\n绘制特征重要性...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        feature_names = load_iris().feature_names

        # 决策树特征重要性
        if '决策树' in self.results:
            dt_model = self.results['决策树']['model']
            importances_dt = dt_model.feature_importances_

            axes[0].bar(feature_names, importances_dt)
            axes[0].set_title('决策树 - 特征重要性')
            axes[0].set_xlabel('特征')
            axes[0].set_ylabel('重要性')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)

        # 随机森林特征重要性
        if '随机森林' in self.results:
            rf_model = self.results['随机森林']['model']
            importances_rf = rf_model.feature_importances_

            axes[1].bar(feature_names, importances_rf, color='green', alpha=0.7)
            axes[1].set_title('随机森林 - 特征重要性')
            axes[1].set_xlabel('特征')
            axes[1].set_ylabel('重要性')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config.get_plot_path('feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_decision_tree(self):
        """可视化决策树"""
        print("\n绘制决策树结构...")

        plt.figure(figsize=(20, 10))
        dt_model = self.results['决策树']['model']
        feature_names = load_iris().feature_names
        target_names = load_iris().target_names

        plot_tree(dt_model,
                 feature_names=feature_names,
                 class_names=target_names,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=3)  # 限制深度以便观察

        plt.title('决策树结构（深度限制为3）')
        plt.savefig(config.get_plot_path('decision_tree.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def print_detailed_report(self):
        """打印详细报告"""
        print("\n" + "="*60)
        print("5. 详细评估报告")
        print("="*60)

        target_names = load_iris().target_names

        for name, result in self.results.items():
            print(f"\n{'='*20} {name} {'='*20}")

            # 分类报告
            print("\n分类报告:")
            print(classification_report(self.y_test, result['y_pred'],
                                      target_names=target_names))

            # 混淆矩阵分析
            print("\n混淆矩阵分析:")
            cm = confusion_matrix(self.y_test, result['y_pred'])

            print("正确分类的样本数:", np.diag(cm).sum())
            print("错误分类的样本数:", cm.sum() - np.diag(cm).sum())

            # 每个类别的详细分析
            for i, class_name in enumerate(target_names):
                true_positive = cm[i, i]
                false_positive = cm[:, i].sum() - true_positive
                false_negative = cm[i, :].sum() - true_positive
                true_negative = cm.sum() - true_positive - false_positive - false_negative

                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"\n{class_name}:")
                print(f"  精确率: {precision:.4f}")
                print(f"  召回率: {recall:.4f}")
                print(f"  F1分数: {f1:.4f}")
                print(f"  正确分类: {true_positive}, 错误分类: {false_negative}")

    def create_comparison_table(self):
        """创建对比表格"""
        print("\n" + "="*60)
        print("6. 模型对比总表")
        print("="*60)

        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                '模型': name,
                '测试集准确率': f"{result['accuracy']:.4f}",
                '交叉验证准确率': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                '训练样本数': len(self.y_train),
                '测试样本数': len(self.y_test),
                '支持概率': '是' if result['y_pred_proba'] is not None else '否'
            })

        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))

        # 保存对比表
        csv_path = config.get_report_path('model_comparison.csv')
        df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n对比表已保存为 '{csv_path}'")

    def demonstrate_prediction(self):
        """演示单个样本预测"""
        print("\n" + "="*50)
        print("7. 单个样本预测演示")
        print("="*50)

        # 选择一个测试样本
        sample_idx = 0
        sample_X = self.X_test[sample_idx:sample_idx+1]
        sample_y = self.y_test[sample_idx]
        sample_X_scaled = self.X_test_scaled[sample_idx:sample_idx+1]

        feature_names = load_iris().feature_names
        target_names = load_iris().target_names

        print(f"测试样本特征值:")
        for i, (name, value) in enumerate(zip(feature_names, sample_X[0])):
            print(f"  {name}: {value:.2f}")

        print(f"\n真实标签: {target_names[sample_y]}")

        print(f"\n各模型预测结果:")
        print("-" * 40)

        for name, result in self.results.items():
            model = result['model']

            # 预测
            if name in ['逻辑回归', 'SVM (RBF)', 'KNN']:
                prediction = model.predict(sample_X_scaled)[0]
                probabilities = model.predict_proba(sample_X_scaled)[0] if hasattr(model, 'predict_proba') else None
            else:
                prediction = model.predict(sample_X)[0]
                probabilities = model.predict_proba(sample_X)[0] if hasattr(model, 'predict_proba') else None

            print(f"{name}:")
            print(f"  预测类别: {target_names[prediction]}")

            if probabilities is not None:
                print(f"  预测概率:")
                for i, (class_name, prob) in enumerate(zip(target_names, probabilities)):
                    print(f"    {class_name}: {prob:.4f}")

            print(f"  预测正确: {'是' if prediction == sample_y else '否'}")
            print()

    def run_complete_analysis(self):
        """运行完整的对比分析"""
        print("开始分类算法对比分析...")

        # 1. 数据准备
        self.load_and_prepare_data()

        # 2. 初始化模型
        self.initialize_models()

        # 3. 训练和评估
        self.train_and_evaluate_models()

        # 4. 可视化结果
        self.visualize_results()

        # 5. 详细报告
        self.print_detailed_report()

        # 6. 对比表格
        self.create_comparison_table()

        # 7. 预测演示
        self.demonstrate_prediction()

        print("\n" + "="*60)
        print("分类算法对比分析完成！")
        print("="*60)

# 主函数
def main():
    """主函数"""
    print("="*60)
    print("分类算法对比教程")
    print("="*60)

    # 创建分析器
    analyzer = ClassificationComparison()

    # 运行完整分析
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()