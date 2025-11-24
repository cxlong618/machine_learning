"""
精简版工具函数模块
只保留可靠的路径管理功能
移除有问题的字体配置
"""
import os
from pathlib import Path

class MLPathManager:
    """机器学习项目路径管理器"""

    def __init__(self):
        self.setup_paths()

    def setup_paths(self):
        """设置项目路径"""
        # 项目根目录
        self.root_dir = Path(__file__).parent
        # 输出目录
        self.output_dir = self.root_dir / "outputs"
        self.plots_dir = self.output_dir / "plots"
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"

        # 创建所有必要的目录
        for directory in [self.output_dir, self.plots_dir, self.models_dir,
                         self.reports_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_plot_path(self, filename):
        """获取图片保存路径"""
        return str(self.plots_dir / filename)

    def get_model_path(self, filename):
        """获取模型保存路径"""
        return str(self.models_dir / filename)

    def get_report_path(self, filename):
        """获取报告保存路径"""
        return str(self.reports_dir / filename)

    def get_data_path(self, filename):
        """获取数据保存路径"""
        return str(self.data_dir / filename)

    def print_paths(self):
        """打印所有路径信息"""
        print("📁 项目路径配置:")
        print(f"  根目录: {self.root_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  图片目录: {self.plots_dir}")
        print(f"  模型目录: {self.models_dir}")
        print(f"  报告目录: {self.reports_dir}")
        print(f"  数据目录: {self.data_dir}")

# 创建全局路径管理实例
path_manager = MLPathManager()

# 为了向后兼容，保留原有的变量名
config = path_manager

def ensure_chinese_font():
    """中文字体设置函数 - 简化版本"""
    import matplotlib.pyplot as plt
    import platform

    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    print(f"✅ 中文字体已设置 (系统: {system})")

# 向后兼容的函数别名
def setup_chinese_font():
    """设置中文字体的别名函数"""
    ensure_chinese_font()