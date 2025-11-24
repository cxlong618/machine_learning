# 📊 机器学习代码示例

本目录包含完整的机器学习算法示例，涵盖了监督学习、无监督学习等核心概念。

## 📁 目录结构

```
代码示例/
├── 📄 README.md                   # 本说明文件
├── 📄 requirements.txt            # 依赖包列表
├── 📄 utils.py                    # 工具函数和配置
├── 📄 test_setup.py              # 环境测试脚本
├── 📄 安装说明.md                 # 详细安装指南
├── 🐍 线性回归示例.py              # 回归算法示例
├── 🐍 分类算法对比.py              # 分类算法对比
├── 🐍 聚类算法实践.py              # 聚类算法实践
└── 📂 outputs/                    # 输出文件目录
    ├── 📂 plots/                   # 图片输出
    ├── 📂 models/                  # 模型保存
    ├── 📂 reports/                 # 报告和数据
    └── 📂 data/                    # 数据文件
```

## 🚀 快速开始

### 1. 环境配置
```bash
# 测试环境
python test_setup.py

# 如果测试失败，安装依赖
pip install -r requirements.txt
```

### 2. 运行示例
```bash
# 基础回归算法
python 线性回归示例.py

# 分类算法对比
python 分类算法对比.py

# 聚类算法实践
python 聚类算法实践.py
```

## 📋 示例说明

### 1. 线性回归示例.py
**功能**: 完整的线性回归实现和演示

**包含内容**:
- 一元和多元线性回归
- 模型评估和可视化
- 正则化方法对比
- 残差分析

**输出文件**:
- `linear_regression_results.png` - 回归结果可视化
- 详细的分析报告

### 2. 分类算法对比.py
**功能**: 多种分类算法的性能对比

**包含内容**:
- 5种分类算法：逻辑回归、决策树、随机森林、SVM、KNN
- 交叉验证和模型评估
- 特征重要性分析
- 混淆矩阵可视化
- 决策树结构图

**输出文件**:
- `classification_comparison.png` - 算法对比图
- `feature_importance.png` - 特征重要性
- `decision_tree.png` - 决策树结构
- `model_comparison.csv` - 性能对比表

### 3. 聚类算法实践.py
**功能**: 聚类算法的全面实践

**包含内容**:
- K-Means、层次聚类、DBSCAN算法
- 最优K值选择方法
- PCA降维可视化
- 客户分群实战案例

**输出文件**:
- `clustering_results.png` - 聚类结果对比
- `dendrogram.png` - 层次聚类树状图
- `pca_visualization.png` - PCA可视化
- `customer_segmentation.png` - 客户分群图
- `clustering_comparison.csv` - 性能对比表

## 🛠️ 核心功能

### 字体配置
自动配置中文字体显示，支持：
- Windows: SimHei, Microsoft YaHei
- macOS: PingFang SC, Arial Unicode MS
- Linux: WenQuanYi Micro Hei

### 输出管理
所有输出文件自动保存到 `outputs/` 目录：
- 图片: `outputs/plots/`
- 模型: `outputs/models/`
- 报告: `outputs/reports/`
- 数据: `outputs/data/`

### 配置管理
`utils.py` 提供统一的配置管理：
- 路径配置
- 字体设置
- 图形样式
- DPI和质量设置

## 🎯 学习路径建议

### 初学者 (1-2周)
1. 运行 `test_setup.py` 确认环境
2. 学习 `线性回归示例.py` 了解回归问题
3. 学习 `分类算法对比.py` 了解分类问题

### 进阶学习者 (3-4周)
1. 深入研究聚类算法实践.py
2. 修改参数进行实验
3. 尝试用自己的数据集

### 专业开发者 (5-6周)
1. 结合不同算法构建解决方案
2. 优化代码性能
3. 开发自己的机器学习项目

## 🔧 自定义配置

### 修改输出路径
在 `utils.py` 中修改 `MLConfig` 类：

```python
def setup_paths(self):
    # 自定义输出目录
    self.custom_output_dir = Path("/path/to/your/output")
```

### 修改图形样式
```python
# 设置不同的绘图风格
plt.style.use('seaborn-darkgrid')
# 或者使用ggplot风格
plt.style.use('ggplot')
```

### 添加新的算法
可以基于现有代码添加新的机器学习算法：

```python
# 添加新的分类算法
from sklearn.ensemble import GradientBoostingClassifier

# 在模型字典中添加
'Gradient Boosting': GradientBoostingClassifier(random_state=42)
```

## 📊 性能基准

在我的测试环境（Python 3.9, Windows 10）下的运行时间：

| 示例 | 数据规模 | 运行时间 | 内存使用 |
|------|----------|----------|----------|
| 线性回归 | 100样本 | <5秒 | <100MB |
| 分类对比 | 鸢尾花数据 | <10秒 | <200MB |
| 聚类实践 | 多个数据集 | <30秒 | <500MB |

## 🐛 常见问题

### Q: 中文字体显示为方框
**A**: 运行 `python test_setup.py` 测试字体，如果失败请：
1. Windows: 安装中文字体包
2. macOS: 系统自带，检查代码
3. Linux: `sudo apt-get install fonts-wqy-microhei`

### Q: 图片保存失败
**A**: 检查 `outputs/` 目录权限，确保有写入权限

### Q: 某些算法运行缓慢
**A**:
1. 减少数据集大小用于测试
2. 使用更快的算法替代
3. 考虑使用GPU加速

### Q: ImportError: No module named 'xxx'
**A**:
```bash
pip install xxx
# 或
pip install -r requirements.txt
```

## 📈 扩展建议

### 添加深度学习示例
- 使用 TensorFlow/Keras 实现神经网络
- CNN 图像分类示例
- RNN 时间序列预测

### 添加强化学习示例
- 使用 Gym 环境
- Q-learning 算法实现
- 简单游戏 AI

### 添加自然语言处理示例
- 文本分类
- 情感分析
- 词向量训练

## 🤝 贡献指南

欢迎贡献代码和建议：

1. Fork 项目
2. 创建特性分支
3. 提交改动
4. 发起 Pull Request

### 代码规范
- 使用 PEP 8 风格
- 添加详细的注释
- 包含必要的测试

## 📞 支持

如果遇到问题：
1. 查看 `安装说明.md` 获取详细安装指导
2. 运行 `test_setup.py` 诊断环境问题
3. 在 GitHub Issues 中报告问题
4. 查看相关技术文档

---

*祝您学习愉快！通过这些示例，您将掌握机器学习的核心概念和实践技能。* 🚀