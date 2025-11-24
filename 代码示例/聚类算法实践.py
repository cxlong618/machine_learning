"""
聚类算法实践
包含：K-Means、层次聚类、DBSCAN、PCA降维
使用生成数据和真实数据演示
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from utils import config, ensure_chinese_font

# 应用字体修复
ensure_chinese_font()
print("✅ 已加载配置，中文字体已设置")

class ClusteringPractice:
    def __init__(self):
        self.datasets = {}
        self.results = {}

    def generate_datasets(self):
        """生成多种聚类测试数据集"""
        print("="*50)
        print("1. 生成聚类测试数据集")
        print("="*50)

        # 1. 球形簇数据
        X1, y1_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0,
                                 random_state=42, center_box=(-10, 10))
        self.datasets['blobs'] = {'X': X1, 'y_true': y1_true, 'name': '球形簇'}

        # 2. 月牙形数据
        X2, y2_true = make_moons(n_samples=300, noise=0.1, random_state=42)
        self.datasets['moons'] = {'X': X2, 'y_true': y2_true, 'name': '月牙形'}

        # 3. 圆形数据
        X3, y3_true = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
        self.datasets['circles'] = {'X': X3, 'y_true': y3_true, 'name': '同心圆'}

        # 4. 不同的密度和方差
        X4, y4_true = make_blobs(n_samples=300, centers=3,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=42)
        self.datasets['varied'] = {'X': X4, 'y_true': y4_true, 'name': '不同密度'}

        # 5. 真实数据集 - 鸢尾花
        iris = load_iris()
        X5 = iris.data[:, :2]  # 只使用前两个特征便于可视化
        y5_true = iris.target
        self.datasets['iris'] = {'X': X5, 'y_true': y5_true, 'name': '鸢尾花（前两维）'}

        # 6. 高维鸢尾花数据
        X6 = iris.data
        self.datasets['iris_4d'] = {'X': X6, 'y_true': y5_true, 'name': '鸢尾花（四维）'}

        print("已生成的数据集:")
        for name, info in self.datasets.items():
            print(f"- {info['name']} ({name}): {info['X'].shape[0]} 样本, {info['X'].shape[1]} 维")

    def find_optimal_k(self, X, max_k=10):
        """使用肘部法则和轮廓系数寻找最优K值"""
        print("\n寻找最优K值...")

        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))

        # 肘部法则寻找最优K
        # 计算肘部点
        distances = []
        for i in range(1, len(wcss) - 1):
            distance = abs(wcss[i-1] + wcss[i+1] - 2*wcss[i])
            distances.append(distance)
        elbow_k = np.argmax(distances) + 2  # +2 because k starts from 2

        # 轮廓系数寻找最优K
        best_silhouette_k = np.argmax(silhouette_scores) + 2

        return {
            'elbow_k': elbow_k,
            'silhouette_k': best_silhouette_k,
            'wcss': wcss,
            'silhouette_scores': silhouette_scores
        }

    def apply_clustering_algorithms(self):
        """应用不同聚类算法"""
        print("\n" + "="*50)
        print("2. 应用聚类算法")
        print("="*50)

        for dataset_name, dataset_info in self.datasets.items():
            X = dataset_info['X']
            y_true = dataset_info['y_true']
            dataset_name_cn = dataset_info['name']

            print(f"\n处理数据集: {dataset_name_cn}")
            print("-" * 30)

            self.results[dataset_name] = {}

            # 数据标准化（除可视化数据集外）
            if dataset_name in ['iris_4d']:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            # 1. K-Means聚类
            if dataset_name in ['blobs', 'varied', 'iris', 'iris_4d']:
                # 寻找最优K值
                optimal_info = self.find_optimal_k(X_scaled)
                print(f"肘部法则建议K: {optimal_info['elbow_k']}")
                print(f"轮廓系数建议K: {optimal_info['silhouette_k']}")

                # 使用轮廓系数建议的K值
                k = optimal_info['silhouette_k']
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_kmeans = kmeans.fit_predict(X_scaled)

                self.results[dataset_name]['kmeans'] = {
                    'labels': labels_kmeans,
                    'n_clusters': k,
                    'centers': kmeans.cluster_centers_,
                    'inertia': kmeans.inertia_,
                    'optimal_info': optimal_info
                }

                print(f"K-Means聚类: {k} 个簇, 轮廓系数: {silhouette_score(X_scaled, labels_kmeans):.4f}")

            # 2. 层次聚类
            n_clusters = len(np.unique(y_true))
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels_hierarchical = hierarchical.fit_predict(X_scaled)

            self.results[dataset_name]['hierarchical'] = {
                'labels': labels_hierarchical,
                'n_clusters': n_clusters
            }

            print(f"层次聚类: {n_clusters} 个簇, 轮廓系数: {silhouette_score(X_scaled, labels_hierarchical):.4f}")

            # 3. DBSCAN聚类
            # 对于不同数据集使用不同参数
            if dataset_name in ['blobs', 'varied']:
                eps = 0.5
                min_samples = 5
            elif dataset_name in ['moons', 'circles']:
                eps = 0.15
                min_samples = 5
            else:  # iris datasets
                eps = 0.5
                min_samples = 5

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels_dbscan = dbscan.fit_predict(X_scaled)

            # 统计聚类结果
            n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
            n_noise = list(labels_dbscan).count(-1)

            self.results[dataset_name]['dbscan'] = {
                'labels': labels_dbscan,
                'n_clusters': n_clusters_dbscan,
                'n_noise': n_noise,
                'eps': eps,
                'min_samples': min_samples
            }

            # 计算轮廓系数（忽略噪声点）
            if n_clusters_dbscan > 1 and len(set(labels_dbscan[labels_dbscan != -1])) > 1:
                valid_indices = labels_dbscan != -1
                if len(set(labels_dbscan[valid_indices])) > 1:
                    sil_score = silhouette_score(X_scaled[valid_indices], labels_dbscan[valid_indices])
                    print(f"DBSCAN聚类: {n_clusters_dbscan} 个簇, {n_noise} 个噪声点, 轮廓系数: {sil_score:.4f}")
                else:
                    print(f"DBSCAN聚类: {n_clusters_dbscan} 个簇, {n_noise} 个噪声点, 无法计算轮廓系数")
            else:
                print(f"DBSCAN聚类: {n_clusters_dbscan} 个簇, {n_noise} 个噪声点")

    def evaluate_clustering(self, X, labels):
        """评估聚类结果"""
        if len(set(labels)) <= 1:
            return None

        # 轮廓系数
        silhouette = silhouette_score(X, labels)

        # Calinski-Harabasz指数
        ch_score = calinski_harabasz_score(X, labels)

        # Davies-Bouldin指数
        db_score = davies_bouldin_score(X, labels)

        return {
            'silhouette': silhouette,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        }

    def visualize_clustering_results(self):
        """可视化聚类结果"""
        print("\n" + "="*50)
        print("3. 可视化聚类结果")
        print("="*50)

        # 创建大图
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        axes = axes.ravel()

        plot_idx = 0

        for dataset_name, dataset_info in self.datasets.items():
            X = dataset_info['X']
            y_true = dataset_info['y_true']
            dataset_name_cn = dataset_info['name']

            # 聚类结果
            dataset_results = self.results[dataset_name]

            # 1. 真实标签
            ax = axes[plot_idx]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', alpha=0.7)
            ax.set_title(f'{dataset_name_cn}\n真实标签')
            ax.set_xlabel('特征 1')
            ax.set_ylabel('特征 2')
            if len(np.unique(y_true)) <= 10:
                plt.colorbar(scatter, ax=ax)
            plot_idx += 1

            # 2. K-Means结果（如果存在）
            if 'kmeans' in dataset_results:
                ax = axes[plot_idx]
                kmeans_result = dataset_results['kmeans']
                scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans_result['labels'], cmap='tab10', alpha=0.7)
                # 绘制聚类中心
                if len(kmeans_result['centers'][0]) == 2:  # 只在2D时绘制中心
                    ax.scatter(kmeans_result['centers'][:, 0], kmeans_result['centers'][:, 1],
                             c='red', marker='x', s=200, linewidths=3)
                ax.set_title(f'K-Means\n(k={kmeans_result["n_clusters"]})')
                ax.set_xlabel('特征 1')
                ax.set_ylabel('特征 2')
                if len(np.unique(kmeans_result['labels'])) <= 10:
                    plt.colorbar(scatter, ax=ax)
                plot_idx += 1

            # 3. 层次聚类结果
            if 'hierarchical' in dataset_results:
                ax = axes[plot_idx]
                hierarchical_result = dataset_results['hierarchical']
                scatter = ax.scatter(X[:, 0], X[:, 1], c=hierarchical_result['labels'], cmap='tab10', alpha=0.7)
                ax.set_title(f'层次聚类\n(k={hierarchical_result["n_clusters"]})')
                ax.set_xlabel('特征 1')
                ax.set_ylabel('特征 2')
                if len(np.unique(hierarchical_result['labels'])) <= 10:
                    plt.colorbar(scatter, ax=ax)
                plot_idx += 1

            # 4. DBSCAN结果
            if 'dbscan' in dataset_results:
                ax = axes[plot_idx]
                dbscan_result = dataset_results['dbscan']
                labels = dbscan_result['labels']

                # 为噪声点使用特殊颜色
                unique_labels = set(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]  # 黑色表示噪声

                    class_member_mask = (labels == k)
                    ax.scatter(X[class_member_mask, 0], X[class_member_mask, 1],
                             c=[col], alpha=0.7, s=50)

                ax.set_title(f'DBSCAN\n(簇={dbscan_result["n_clusters"]}, 噪声={dbscan_result["n_noise"]})')
                ax.set_xlabel('特征 1')
                ax.set_ylabel('特征 2')
                plot_idx += 1

            # 5. K值选择图（如果有K-Means结果）
            if 'kmeans' in dataset_results:
                ax = axes[plot_idx]
                optimal_info = dataset_results['kmeans']['optimal_info']

                # WCSS肘部图
                ax2 = ax.twinx()
                k_range = range(2, len(optimal_info['wcss']) + 2)

                # 绘制WCSS
                line1 = ax.plot(k_range, optimal_info['wcss'], 'b-o', label='WCSS')
                ax.set_xlabel('K值')
                ax.set_ylabel('WCSS', color='b')
                ax.tick_params(axis='y', labelcolor='b')

                # 绘制轮廓系数
                line2 = ax2.plot(k_range, optimal_info['silhouette_scores'], 'r-o', label='轮廓系数')
                ax2.set_ylabel('轮廓系数', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                # 标记最优K值
                ax.axvline(x=optimal_info['elbow_k'], color='b', linestyle='--', alpha=0.7)
                ax2.axvline(x=optimal_info['silhouette_k'], color='r', linestyle='--', alpha=0.7)

                ax.set_title('K值选择')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(config.get_plot_path('clustering_results.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_dendrogram(self):
        """绘制层次聚类的树状图"""
        print("\n绘制层次聚类树状图...")

        # 选择一个数据集展示树状图
        dataset_name = 'blobs'
        X = self.datasets[dataset_name]['X']

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 计算链接矩阵
        linkage_matrix = linkage(X_scaled, method='ward')

        # 绘制树状图
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix,
                  truncate_mode='lastp',  # 只显示最后p个簇的合并
                  p=12,
                  show_leaf_counts=True,
                  leaf_rotation=90.,
                  leaf_font_size=12.,
                  show_contracted=True)

        plt.title('层次聚类树状图')
        plt.xlabel('样本索引或簇大小')
        plt.ylabel('距离')
        plt.tight_layout()
        plt.savefig(config.get_plot_path('dendrogram.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def pca_visualization(self):
        """PCA降维可视化"""
        print("\n" + "="*50)
        print("4. PCA降维可视化")
        print("="*50)

        # 对高维数据应用PCA
        if 'iris_4d' in self.datasets:
            X = self.datasets['iris_4d']['X']
            y_true = self.datasets['iris_4d']['y_true']

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA降维到2D
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # 聚类
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # 可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 原始标签
            scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='tab10', alpha=0.7)
            ax1.set_title('PCA可视化 - 真实标签')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)')
            plt.colorbar(scatter1, ax=ax1)

            # 聚类结果
            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
            ax2.set_title('PCA可视化 - K-Means聚类')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)')
            plt.colorbar(scatter2, ax=ax2)

            plt.tight_layout()
            plt.savefig(config.get_plot_path('pca_visualization.png'), dpi=300, bbox_inches='tight')
            plt.show()

            print(f"PCA解释方差比例: {pca.explained_variance_ratio_}")
            print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.1%}")

    def performance_comparison(self):
        """性能对比分析"""
        print("\n" + "="*50)
        print("5. 算法性能对比")
        print("="*50)

        comparison_data = []

        for dataset_name, dataset_info in self.datasets.items():
            X = dataset_info['X']
            dataset_name_cn = dataset_info['name']
            dataset_results = self.results[dataset_name]

            # 数据标准化
            if dataset_name in ['iris_4d']:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            for algorithm, result in dataset_results.items():
                if algorithm == 'kmeans' and 'optimal_info' in result:
                    # 对于K-Means，使用轮廓系数建议的K值
                    labels = result['labels']
                else:
                    labels = result['labels']

                # 评估聚类结果（忽略噪声点）
                if -1 in labels:  # DBSCAN有噪声点
                    valid_indices = labels != -1
                    if len(np.unique(labels[valid_indices])) > 1:
                        evaluation = self.evaluate_clustering(X_scaled[valid_indices], labels[valid_indices])
                    else:
                        evaluation = None
                else:
                    evaluation = self.evaluate_clustering(X_scaled, labels)

                if evaluation:
                    comparison_data.append({
                        '数据集': dataset_name_cn,
                        '算法': algorithm,
                        '簇数量': result.get('n_clusters', len(set(labels))),
                        '轮廓系数': f"{evaluation['silhouette']:.4f}",
                        'Calinski-Harabasz': f"{evaluation['calinski_harabasz']:.2f}",
                        'Davies-Bouldin': f"{evaluation['davies_bouldin']:.4f}"
                    })

        # 创建对比表格
        df_comparison = pd.DataFrame(comparison_data)

        print("算法性能对比表:")
        print(df_comparison.to_string(index=False))

        # 保存对比表
        csv_path = config.get_report_path('clustering_comparison.csv')
        df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n对比表已保存为 '{csv_path}'")

        # 按数据集分组展示最佳算法
        print("\n各数据集的最佳算法（按轮廓系数）:")
        print("-" * 50)

        for dataset_name in df_comparison['数据集'].unique():
            dataset_subset = df_comparison[df_comparison['数据集'] == dataset_name]
            best_row = dataset_subset.loc[dataset_subset['轮廓系数'].astype(float).idxmax()]
            print(f"{dataset_name}: {best_row['算法']} (轮廓系数: {best_row['轮廓系数']})")

    def demonstrate_custom_clustering(self):
        """演示自定义聚类应用"""
        print("\n" + "="*50)
        print("6. 自定义聚类应用：客户分群")
        print("="*50)

        # 生成模拟客户数据
        np.random.seed(42)
        n_customers = 500

        # 客户特征：年龄、年收入、消费评分
        ages = np.random.normal(35, 12, n_customers)
        incomes = np.random.lognormal(10.5, 0.5, n_customers)  # 对数正态分布
        scores = np.random.uniform(1, 100, n_customers)

        # 创建不同类型的客户群
        # 年轻高收入群体
        ages[:100] = np.random.normal(28, 5, 100)
        incomes[:100] = np.random.lognormal(11, 0.3, 100)
        scores[:100] = np.random.uniform(60, 95, 100)

        # 中年中等收入群体
        ages[100:250] = np.random.normal(45, 8, 150)
        incomes[100:250] = np.random.lognormal(10.5, 0.3, 150)
        scores[100:250] = np.random.uniform(40, 70, 150)

        # 年长低收入群体
        ages[250:] = np.random.normal(60, 10, 250)
        incomes[250:] = np.random.lognormal(10, 0.3, 250)
        scores[250:] = np.random.uniform(20, 60, 250)

        # 确保数值合理
        ages = np.clip(ages, 18, 80)
        incomes = np.clip(incomes, 10000, 200000)
        scores = np.clip(scores, 1, 100)

        # 创建特征矩阵
        X_customers = np.column_stack([ages, incomes, scores])
        feature_names = ['年龄', '年收入', '消费评分']

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_customers)

        # 寻找最优K值
        optimal_info = self.find_optimal_k(X_scaled, max_k=8)
        k = optimal_info['silhouette_k']

        # K-Means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # 分析聚类结果
        print(f"客户分群结果：{k} 个客户群")
        print(f"整体轮廓系数: {silhouette_score(X_scaled, labels):.4f}")

        # 分析每个客户群的特征
        df_customers = pd.DataFrame(X_customers, columns=feature_names)
        df_customers['cluster'] = labels

        print("\n各客户群特征分析:")
        print("-" * 60)

        for cluster_id in range(k):
            cluster_data = df_customers[df_customers['cluster'] == cluster_id]
            print(f"\n客户群 {cluster_id + 1} ({len(cluster_data)} 位客户):")
            print(f"  平均年龄: {cluster_data['年龄'].mean():.1f} 岁")
            print(f"  平均年收入: {cluster_data['年收入'].mean():.0f} 元")
            print(f"  平均消费评分: {cluster_data['消费评分'].mean():.1f}")

            # 客户群特征描述
            age_desc = "年轻" if cluster_data['年龄'].mean() < 35 else "中年" if cluster_data['年龄'].mean() < 55 else "年长"
            income_desc = "低" if cluster_data['年收入'].mean() < 50000 else "中等" if cluster_data['年收入'].mean() < 100000 else "高"
            score_desc = "低" if cluster_data['消费评分'].mean() < 40 else "中等" if cluster_data['消费评分'].mean() < 70 else "高"

            print(f"  特征: {age_desc}客户, {income_desc}收入, {score_desc}消费评分")

        # 可视化客户分群
        fig = plt.figure(figsize=(15, 5))

        # 年龄 vs 收入
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(df_customers['年龄'], df_customers['年收入'], c=df_customers['cluster'], cmap='tab10', alpha=0.7)
        ax1.set_xlabel('年龄')
        ax1.set_ylabel('年收入')
        ax1.set_title('年龄 vs 年收入')
        plt.colorbar(scatter, ax=ax1)

        # 年龄 vs 消费评分
        ax2 = fig.add_subplot(132)
        scatter = ax2.scatter(df_customers['年龄'], df_customers['消费评分'], c=df_customers['cluster'], cmap='tab10', alpha=0.7)
        ax2.set_xlabel('年龄')
        ax2.set_ylabel('消费评分')
        ax2.set_title('年龄 vs 消费评分')
        plt.colorbar(scatter, ax=ax2)

        # 收入 vs 消费评分
        ax3 = fig.add_subplot(133)
        scatter = ax3.scatter(df_customers['年收入'], df_customers['消费评分'], c=df_customers['cluster'], cmap='tab10', alpha=0.7)
        ax3.set_xlabel('年收入')
        ax3.set_ylabel('消费评分')
        ax3.set_title('年收入 vs 消费评分')
        plt.colorbar(scatter, ax=ax3)

        plt.tight_layout()
        plt.savefig(config.get_plot_path('customer_segmentation.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_practice(self):
        """运行完整的聚类实践"""
        print("开始聚类算法实践...")

        # 1. 生成数据集
        self.generate_datasets()

        # 2. 应用聚类算法
        self.apply_clustering_algorithms()

        # 3. 可视化结果
        self.visualize_clustering_results()

        # 4. 绘制树状图
        self.plot_dendrogram()

        # 5. PCA可视化
        self.pca_visualization()

        # 6. 性能对比
        self.performance_comparison()

        # 7. 自定义应用
        self.demonstrate_custom_clustering()

        print("\n" + "="*60)
        print("聚类算法实践完成！")
        print("="*60)

# 主函数
def main():
    """主函数"""
    print("="*60)
    print("聚类算法实践教程")
    print("="*60)

    # 创建实践对象
    practice = ClusteringPractice()

    # 运行完整实践
    practice.run_complete_practice()

if __name__ == "__main__":
    main()