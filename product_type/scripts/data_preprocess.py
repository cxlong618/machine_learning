#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本（整合版）
将Excel数据转换为CSV格式并验证，支持数据重新分层分割
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
import json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """
    统一的数据清洗函数

    Args:
        df: 原始数据框
        required_columns: 必需的列名列表

    Returns:
        清洗后的数据框
    """
    logger.info("开始数据清洗...")

    original_length = len(df)

    # 去除空值和重复值
    df = df.dropna(subset=required_columns)
    df = df.drop_duplicates(subset=['product_name'])

    logger.info(f"数据清洗: {original_length} -> {len(df)} 行")

    # 数据类型转换
    for col in required_columns:
        df[col] = df[col].astype(str).str.strip()

    # 过滤产品名称长度
    df = df[df['product_name'].str.len() >= 2]
    df = df[df['product_name'].str.len() <= 100]

    # 过滤空字符串
    for col in required_columns:
        df = df[df[col] != '']

    logger.info(f"长度和空值过滤后: {len(df)} 行")

    return df


def analyze_data_quality(df: pd.DataFrame, required_columns: list):
    """
    分析数据质量并输出统计信息

    Args:
        df: 数据框
        required_columns: 必需的列名列表
    """
    logger.info("数据质量检查...")

    # 检查分布
    for col in required_columns:
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()

        logger.info(f"  {col}: {unique_count} 唯一值, {null_count} 空值")

        if col == 'product_name':
            if len(df) != unique_count:
                logger.warning(f"产品名称存在重复: {len(df)} 行, {unique_count} 唯一")

    # 标签统计
    for col in ['standard_name', 'level1', 'level2', 'level3']:
        value_counts = df[col].value_counts()
        logger.info(f"\n{col} 分布 (Top 10):")
        for i, (value, count) in enumerate(value_counts.head(10).items(), 1):
            logger.info(f"  {i:2d}. {value}: {count} ({count/len(df)*100:.1f}%)")

    # 文本长度分析
    df['text_length'] = df['product_name'].str.len()
    logger.info(f"\n产品名称长度统计:")
    logger.info(f"  平均长度: {df['text_length'].mean():.2f}")
    logger.info(f"  最短长度: {df['text_length'].min()}")
    logger.info(f"  最长长度: {df['text_length'].max()}")
    logger.info(f"  标准差: {df['text_length'].std():.2f}")


def stratified_split(df: pd.DataFrame, output_dir: str = "data",
                    train_ratio: float = 0.8, val_ratio: float = 0.1,
                    min_samples_per_class: int = 5, random_state: int = 42,
                    stratify_col: str = 'standard_name'):
    """
    分层分割数据集

    Args:
        df: 输入数据框
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        min_samples_per_class: 每个类别最少样本数
        random_state: 随机种子
        stratify_col: 分层依据的列名

    Returns:
        train_df, val_df, test_df
    """
    logger.info("开始分层分割数据集...")

    # 创建输出目录 - 支持相对路径和绝对路径
    if not os.path.isabs(output_dir):
        # 如果是相对路径，则相对于脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 检查类别分布
    class_counts = df[stratify_col].value_counts()
    logger.info(f"类别分布:")
    logger.info(f"  总类别数: {len(class_counts)}")

    # 找出样本数量足够的类别（>=2个样本才能分层分割）
    valid_classes = class_counts[class_counts >= 2].index.tolist()
    valid_df = df[df[stratify_col].isin(valid_classes)].copy()

    # 找出样本不足的类别（只有1个样本），全部放入训练集
    invalid_df = df[~df[stratify_col].isin(valid_classes)].copy()

    logger.info(f"可分层类别(≥2样本): {len(valid_classes)}")
    logger.info(f"单样本类别: {len(class_counts) - len(valid_classes)}")
    logger.info(f"单样本数据: {len(invalid_df)} 行")

    # 对有效数据进行分层分割
    if len(valid_df) > 0:
        # 使用StratifiedShuffleSplit进行分层分割
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_state)

        # 分割出训练集和临时集（验证+测试）
        train_indices, temp_indices = next(splitter.split(valid_df, valid_df[stratify_col]))

        train_from_valid = valid_df.iloc[train_indices]
        temp_df = valid_df.iloc[temp_indices]

        # 再将临时集分层分割为验证集和测试集
        temp_size = 1 - train_ratio
        val_test_ratio = val_ratio / temp_size

        # 检查temp_df中的类别分布，避免单样本类别
        temp_class_counts = temp_df[stratify_col].value_counts()
        temp_valid_classes = temp_class_counts[temp_class_counts >= 2].index.tolist()
        temp_valid_df = temp_df[temp_df[stratify_col].isin(temp_valid_classes)].copy()
        temp_invalid_df = temp_df[~temp_df[stratify_col].isin(temp_valid_classes)].copy()

        if len(temp_valid_df) > 0:
            temp_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_state)
            val_indices, test_indices = next(temp_splitter.split(temp_valid_df, temp_valid_df[stratify_col]))

            val_from_temp = temp_valid_df.iloc[val_indices]
            test_from_temp = temp_valid_df.iloc[test_indices]

            # 将单样本类别平均加入验证集和测试集
            if not temp_invalid_df.empty:
                # 随机分配到验证集和测试集
                mid_point = len(temp_invalid_df) // 2
                val_extra = temp_invalid_df.iloc[:mid_point]
                test_extra = temp_invalid_df.iloc[mid_point:]
            else:
                val_extra = pd.DataFrame()
                test_extra = pd.DataFrame()

            val_df = pd.concat([val_from_temp, val_extra], ignore_index=True)
            test_df = pd.concat([test_from_temp, test_extra], ignore_index=True)
        else:
            # 如果没有可分层的类别，随机分配
            if not temp_df.empty:
                mid_point = len(temp_df) // 2
                val_df = temp_df.iloc[:mid_point]
                test_df = temp_df.iloc[mid_point:]
            else:
                val_df = pd.DataFrame()
                test_df = pd.DataFrame()
    else:
        # 如果没有有效类别，全部放入训练集
        train_from_valid = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

    # 合并样本不足的数据到训练集
    train_df = pd.concat([train_from_valid, invalid_df], ignore_index=True)

    # 随机打乱最终数据
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    if not val_df.empty:
        val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    if not test_df.empty:
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    total_samples = len(df)
    logger.info(f"\n数据集分层分割:")
    logger.info(f"  训练集: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    logger.info(f"  验证集: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
    logger.info(f"  测试集: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")

    # 验证分层效果
    logger.info(f"\n分层效果验证:")
    for dataset_name, dataset_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        if len(dataset_df) > 0:
            unique_classes = dataset_df[stratify_col].nunique()
            logger.info(f"  {dataset_name}: {unique_classes} 个标准名称类别")

            # 显示主要类别分布
            top_classes = dataset_df[stratify_col].value_counts().head(5)
            for class_name, count in top_classes.items():
                logger.info(f"    {class_name}: {count} 样本")

    # 验证训练集包含所有类别
    train_classes = set(train_df[stratify_col].unique())
    val_classes = set(val_df[stratify_col].unique()) if not val_df.empty else set()
    test_classes = set(test_df[stratify_col].unique()) if not test_df.empty else set()

    unknown_in_val = val_classes - train_classes
    unknown_in_test = test_classes - train_classes

    if unknown_in_val:
        logger.warning(f"验证集包含训练集中未出现的类别: {list(unknown_in_val)}")
    if unknown_in_test:
        logger.warning(f"测试集包含训练集中未出现的类别: {list(unknown_in_test)}")

    if not unknown_in_val and not unknown_in_test:
        logger.info("训练集包含所有验证集和测试集的类别")

    # 保存CSV文件
    train_df.to_csv(f"{output_dir}/train.csv", index=False, encoding='utf-8')
    val_df.to_csv(f"{output_dir}/val.csv", index=False, encoding='utf-8')
    test_df.to_csv(f"{output_dir}/test.csv", index=False, encoding='utf-8')

    logger.info(f"\n分层分割完成! 输出文件:")
    logger.info(f"  - {output_dir}/train.csv")
    logger.info(f"  - {output_dir}/val.csv")
    logger.info(f"  - {output_dir}/test.csv")

    return train_df, val_df, test_df


def preprocess_excel_data(excel_path: str, output_dir: str = "data"):
    """
    预处理Excel数据，转换为CSV格式并进行分层分割

    Args:
        excel_path: Excel文件路径
        output_dir: 输出目录
    """
    logger.info(f"开始预处理Excel文件: {excel_path}")

    # 创建输出目录 - 支持相对路径和绝对路径
    if not os.path.isabs(output_dir):
        # 如果是相对路径，则相对于脚本所在目录的上一级目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)  # 获取scripts目录的上一级目录
        output_dir = os.path.join(parent_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 读取Excel文件
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            logger.error(f"读取Excel文件失败: {e}")
            raise

        logger.info(f"原始数据形状: {df.shape}")
        logger.info(f"列名: {list(df.columns)}")

        # 检查必需的列
        required_columns = ['product_name', 'standard_name', 'level1', 'level2', 'level3']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"缺少必需的列: {missing_columns}")
            logger.info(f"可用的列: {list(df.columns)}")

            # 尝试映射列名
            column_mapping = {
                '产品名称': 'product_name',
                '标准名称': 'standard_name',
                '一级分类': 'level1',
                '二级分类': 'level2',
                '三级分类': 'level3'
            }

            # 尝试找到正确的列名
            mapped_columns = {}
            for required_col in required_columns:
                for col in df.columns:
                    if required_col in column_mapping.values():
                        if column_mapping.get(col) == required_col:
                            mapped_columns[col] = required_col
                            break
                    elif any(keyword in str(col) for keyword in required_col.split('_')):
                        mapped_columns[col] = required_col
                        break

            if len(mapped_columns) >= len(required_columns):
                logger.info("找到映射列名，重命名列...")
                df = df.rename(columns=mapped_columns)
                missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"无法找到必需的列: {missing_columns}")

        # 数据清洗
        df = clean_data(df, required_columns)

        # 数据质量分析
        analyze_data_quality(df, required_columns)

        # 保存数据统计
        stats = {
            'total_samples': len(df),
            'unique_products': df['product_name'].nunique(),
            'unique_standard_names': df['standard_name'].nunique(),
            'unique_level1': df['level1'].nunique(),
            'unique_level2': df['level2'].nunique(),
            'unique_level3': df['level3'].nunique(),
            'text_length_stats': {
                'mean': float(df['text_length'].mean()),
                'min': int(df['text_length'].min()),
                'max': int(df['text_length'].max()),
                'std': float(df['text_length'].std())
            },
            'category_distribution': {
                'standard_name': df['standard_name'].value_counts().to_dict(),
                'level1': df['level1'].value_counts().to_dict(),
                'level2': df['level2'].value_counts().to_dict(),
                'level3': df['level3'].value_counts().to_dict()
            }
        }

        with open(f"{output_dir}/data_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

        # 分层分割数据集
        _ = stratified_split(df, output_dir, train_ratio=0.8, val_ratio=0.1, min_samples_per_class=2, random_state=42)

        logger.info("\n预处理完成!")

        return stats

    except Exception as e:
        logger.error(f"预处理失败: {e}")
        raise


def resplit_csv_data(input_csv: str, output_dir: str = "data",
                     train_ratio: float = 0.8, val_ratio: float = 0.1,
                     random_state: int = 42):
    """
    重新对现有CSV文件进行分层分割

    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_state: 随机种子
    """
    logger.info(f"开始重新分层分割: {input_csv}")

    # 加载数据
    df = pd.read_csv(input_csv, encoding='utf-8')
    logger.info(f"原始数据: {len(df)} 行, {df.columns.tolist()}")

    # 检查必需列
    required_columns = ['product_name', 'standard_name', 'level1', 'level2', 'level3']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")

    # 数据清洗
    df = clean_data(df, required_columns)

    # 分层分割
    train_df, val_df, test_df = stratified_split(
        df, output_dir, train_ratio, val_ratio,
        2, random_state  # 固定最小样本数为2
    )

    return train_df, val_df, test_df


def validate_csv_format(csv_path: str):
    """验证CSV格式是否正确"""
    logger.info(f"验证CSV文件: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')

        required_columns = ['product_name', 'standard_name', 'level1', 'level2', 'level3']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"CSV文件缺少必需的列: {missing_columns}")
            return False

        # 检查数据质量
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"发现空值: {null_counts.to_dict()}")

        # 检查行数
        if len(df) < 1000:
            logger.warning(f"数据量较少: {len(df)} 行")

        logger.info("CSV文件验证通过")
        return True

    except Exception as e:
        logger.error(f"CSV文件验证失败: {e}")
        return False


def validate_data_consistency(train_path: str, val_path: str, test_path: str = None):
    """验证数据集一致性"""
    logger.info("验证数据集一致性...")

    # 加载数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    datasets = [
        ("训练集", train_df),
        ("验证集", val_df)
    ]

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        datasets.append(("测试集", test_df))

    # 获取训练集的所有类别
    train_classes = {
        'standard_name': set(train_df['standard_name'].unique()),
        'level1': set(train_df['level1'].unique()),
        'level2': set(train_df['level2'].unique()),
        'level3': set(train_df['level3'].unique()),
    }

    # 检查每个数据集
    all_consistent = True
    for dataset_name, dataset_df in datasets[1:]:  # 跳过训练集
        logger.info(f"\n检查{dataset_name}...")

        for col in ['standard_name', 'level1', 'level2', 'level3']:
            dataset_classes = set(dataset_df[col].unique())
            unknown_classes = dataset_classes - train_classes[col]

            if unknown_classes:
                logger.warning(f"  {col}: {len(unknown_classes)} 个未知类别")
                for unknown_class in list(unknown_classes)[:5]:  # 只显示前5个
                    logger.warning(f"    - {unknown_class}")
                if len(unknown_classes) > 5:
                    logger.warning(f"    ... 还有 {len(unknown_classes)-5} 个")
                all_consistent = False
            else:
                logger.info(f"  {col}: 所有类别都在训练集中")

    if all_consistent:
        logger.info("\n数据集一致性验证通过!")
    else:
        logger.warning("\n发现数据集一致性问题，建议重新分层分割")

    return all_consistent


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='数据预处理脚本（整合版）')
    parser.add_argument('--input', type=str, help='输入Excel或CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--validate_only', action='store_true', help='仅验证现有CSV文件')
    parser.add_argument('--resplit', action='store_true', help='重新对现有CSV文件进行分层分割')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    print("产品分类数据预处理脚本（整合版）")
    print("=" * 60)

    if args.validate_only:
        # 仅验证模式
        csv_files = ['data/train.csv', 'data/val.csv']
        if os.path.exists('data/test.csv'):
            csv_files.append('data/test.csv')

        # 验证格式
        all_format_ok = True
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                if validate_csv_format(csv_file):
                    print(f"[通过] {csv_file} 格式验证通过")
                else:
                    print(f"[失败] {csv_file} 格式验证失败")
                    all_format_ok = False
            else:
                print(f"[警告] {csv_file} 不存在")
                all_format_ok = False

        if all_format_ok and all(os.path.exists(f) for f in csv_files):
            # 验证一致性
            validate_data_consistency('data/train.csv', 'data/val.csv', 'data/test.csv' if 'data/test.csv' in csv_files else None)

    elif args.resplit:
        # 重新分割模式
        if not args.input:
            print("[错误] 重新分割模式需要指定 --input 参数")
            return 1

        if not os.path.exists(args.input):
            print(f"[错误] 输入文件不存在: {args.input}")
            return 1

        try:
            train_df, val_df, test_df = resplit_csv_data(
                args.input,
                args.output_dir,
                args.train_ratio,
                args.val_ratio,
                args.random_state
            )

            print("\n分割统计:")
            print(f"  总样本数: {len(train_df) + len(val_df) + len(test_df):,}")
            print(f"  训练集: {len(train_df):,} 样本")
            print(f"  验证集: {len(val_df):,} 样本")
            print(f"  测试集: {len(test_df):,} 样本")

            print(f"\n训练集类别覆盖:")
            for col in ['standard_name', 'level1', 'level2', 'level3']:
                unique_count = train_df[col].nunique()
                print(f"  {col}: {unique_count} 类")

            print("\n重新分层分割完成! 可以开始训练了。")
            print(f"\n运行训练:")
            print(f"python src/train.py --train_path {args.output_dir}/train.csv --val_path {args.output_dir}/val.csv")

        except Exception as e:
            print(f"\n[错误] 重新分割失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        # 预处理模式（Excel → CSV）
        if not args.input:
            print("[错误] 预处理模式需要指定 --input 参数")
            return 1

        if not os.path.exists(args.input):
            print(f"[错误] 输入文件不存在: {args.input}")
            return 1

        try:
            stats = preprocess_excel_data(args.input, args.output_dir)

            print("\n处理统计:")
            print(f"  总样本数: {stats['total_samples']:,}")
            print(f"  独特产品数: {stats['unique_products']:,}")
            print(f"  标准名称数: {stats['unique_standard_names']}")
            print(f"  一级分类数: {stats['unique_level1']}")
            print(f"  二级分类数: {stats['unique_level2']}")
            print(f"  三级分类数: {stats['unique_level3']}")

            print("\n产品名称长度:")
            length_stats = stats['text_length_stats']
            print(f"  平均: {length_stats['mean']:.2f}")
            print(f"  范围: {length_stats['min']} - {length_stats['max']}")
            print(f"  标准差: {length_stats['std']:.2f}")

            print("\n预处理完成! 可以开始训练了。")
            print(f"\n运行训练:")
            print(f"python src/train.py --train_path {args.output_dir}/train.csv --val_path {args.output_dir}/val.csv")

        except Exception as e:
            print(f"\n[错误] 预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)