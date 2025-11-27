#!/usr/bin/env python3
"""
数据验证和预处理脚本
检查训练集和验证集的标签一致性
"""
import pandas as pd
import json
import logging
from typing import Dict, Set, List
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_data_consistency(train_path: str, val_path: str) -> bool:
    """验证训练集和验证集的标签一致性"""

    logger.info("开始验证数据一致性...")

    # 加载数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    logger.info(f"训练集样本数: {len(train_df)}")
    logger.info(f"验证集样本数: {len(val_df)}")

    # 获取各分类的唯一值
    train_standard = set(train_df['standard_name'].unique())
    val_standard = set(val_df['standard_name'].unique())

    train_level1 = set(train_df['level1'].unique())
    val_level1 = set(val_df['level1'].unique())

    train_level2 = set(train_df['level2'].unique())
    val_level2 = set(val_df['level2'].unique())

    train_level3 = set(train_df['level3'].unique())
    val_level3 = set(val_df['level3'].unique())

    # 检查未知标签
    issues = []

    # 标准名称检查
    unknown_standard = val_standard - train_standard
    if unknown_standard:
        issues.append(f"验证集包含训练集中没有的标准名称 ({len(unknown_standard)}个): {list(unknown_standard)[:5]}...")
        logger.warning(f"未知标准名称: {unknown_standard}")

    # 一级分类检查
    unknown_level1 = val_level1 - train_level1
    if unknown_level1:
        issues.append(f"验证集包含训练集中没有的一级分类 ({len(unknown_level1)}个): {list(unknown_level1)}")
        logger.warning(f"未知一级分类: {unknown_level1}")

    # 二级分类检查
    unknown_level2 = val_level2 - train_level2
    if unknown_level2:
        issues.append(f"验证集包含训练集中没有的二级分类 ({len(unknown_level2)}个): {list(unknown_level2)}")
        logger.warning(f"未知二级分类: {unknown_level2}")

    # 三级分类检查
    unknown_level3 = val_level3 - train_level3
    if unknown_level3:
        issues.append(f"验证集包含训练集中没有的三级分类 ({len(unknown_level3)}个): {list(unknown_level3)}")
        logger.warning(f"未知三级分类: {unknown_level3}")

    if issues:
        logger.error("数据一致性问题:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ 数据一致性检查通过")
        return True


def create_clean_val_dataset(train_path: str, val_path: str, output_path: str = None) -> str:
    """创建只包含训练集中已知标签的干净验证集"""

    if output_path is None:
        output_path = val_path.replace('.csv', '_filtered.csv')

    logger.info("创建过滤后的验证集...")

    # 加载数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 获取训练集的已知标签
    known_standard = set(train_df['standard_name'].unique())
    known_level1 = set(train_df['level1'].unique())
    known_level2 = set(train_df['level2'].unique())
    known_level3 = set(train_df['level3'].unique())

    original_count = len(val_df)

    # 过滤验证集
    filtered_df = val_df[
        val_df['standard_name'].isin(known_standard) &
        val_df['level1'].isin(known_level1) &
        val_df['level2'].isin(known_level2) &
        val_df['level3'].isin(known_level3)
    ].copy()

    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    logger.info(f"原始验证集: {original_count} 样本")
    logger.info(f"过滤后验证集: {filtered_count} 样本")
    logger.info(f"移除样本: {removed_count} ({removed_count/original_count*100:.1f}%)")

    # 保存过滤后的数据
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"过滤后的验证集已保存: {output_path}")

    return output_path


def analyze_data_distribution(train_path: str, val_path: str):
    """分析数据分布"""

    logger.info("分析数据分布...")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print("\n=== 数据分布分析 ===")

    for col in ['standard_name', 'level1', 'level2', 'level3']:
        print(f"\n{col}:")
        print(f"  训练集: {len(train_df[col].unique())} 类")
        print(f"  验证集: {len(val_df[col].unique())} 类")

        # 训练集前10个类别
        train_counts = train_df[col].value_counts().head(10)
        print(f"  训练集主要类别:")
        for label, count in train_counts.items():
            print(f"    {label}: {count}")

    print("\n" + "="*50)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='数据验证和预处理')
    parser.add_argument('--train_path', type=str, required=True, help='训练数据CSV文件')
    parser.add_argument('--val_path', type=str, required=True, help='验证数据CSV文件')
    parser.add_argument('--output_path', type=str, help='过滤后验证集输出路径')
    parser.add_argument('--auto_fix', action='store_true', help='自动创建过滤后的验证集')

    args = parser.parse_args()

    # 检查文件存在性
    if not os.path.exists(args.train_path):
        logger.error(f"训练文件不存在: {args.train_path}")
        return False

    if not os.path.exists(args.val_path):
        logger.error(f"验证文件不存在: {args.val_path}")
        return False

    # 分析数据分布
    analyze_data_distribution(args.train_path, args.val_path)

    # 验证数据一致性
    is_valid = validate_data_consistency(args.train_path, args.val_path)

    if not is_valid:
        if args.auto_fix:
            logger.info("自动修复: 创建过滤后的验证集...")
            output_path = create_clean_val_dataset(args.train_path, args.val_path, args.output_path)
            logger.info(f"请使用过滤后的验证集进行训练:")
            logger.info(f"python src/train.py --train_path {args.train_path} --val_path {output_path}")
        else:
            logger.info("建议:")
            logger.info("1. 使用 --auto_fix 创建过滤后的验证集")
            logger.info("2. 或者手动编辑验证集移除未知标签")
            logger.info("3. 或者确保训练集包含验证集的所有标签")
        return False
    else:
        logger.info("数据验证通过，可以开始训练!")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)