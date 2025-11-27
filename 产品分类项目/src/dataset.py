"""
数据集处理模块
用于加载和预处理产品分类数据
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from modelscope_utils import load_tokenizer
import jieba
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
from collections import Counter
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductDataset(Dataset):
    """
    产品分类数据集
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        is_train: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # 加载数据
        self.data = self._load_data()

        # 创建标签映射
        if is_train:
            self._create_label_mappings()
            self._save_mappings()
        else:
            self._load_mappings()

        logger.info(f"数据集初始化完成: {len(self.data)} 样本")
        logger.info(f"标签分布:")
        logger.info(f"  标准名称: {len(self.standard_mapping)} 类")
        logger.info(f"  一级分类: {len(self.level1_mapping)} 类")
        logger.info(f"  二级分类: {len(self.level2_mapping)} 类")
        logger.info(f"  三级分类: {len(self.level3_mapping)} 类")

    def _load_data(self) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']

            for encoding in encodings:
                try:
                    df = pd.read_csv(self.data_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法读取文件，请检查文件编码")

            # 检查必需的列
            required_columns = ['product_name', 'standard_name', 'level1', 'level2', 'level3']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"缺少必需的列: {missing_columns}")

            # 数据清洗
            df = self._clean_data(df)

            logger.info(f"成功加载数据: {len(df)} 行")
            return df

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 去除空值
        df = df.dropna(subset=['product_name', 'standard_name'])

        # 去除重复项
        df = df.drop_duplicates(subset=['product_name'])

        # 清理产品名称
        df['product_name'] = df['product_name'].astype(str).str.strip()
        df = df[df['product_name'].str.len() >= 2]  # 产品名称至少2个字符
        df = df[df['product_name'].str.len() <= 100]  # 产品名称最多100个字符

        logger.info(f"数据清洗后剩余: {len(df)} 行")
        return df

    def _create_label_mappings(self):
        """创建标签映射"""
        # 创建标签映射字典
        self.standard_mapping = {label: idx for idx, label in enumerate(sorted(self.data['standard_name'].unique()))}
        self.level1_mapping = {label: idx for idx, label in enumerate(sorted(self.data['level1'].unique()))}
        self.level2_mapping = {label: idx for idx, label in enumerate(sorted(self.data['level2'].unique()))}
        self.level3_mapping = {label: idx for idx, label in enumerate(sorted(self.data['level3'].unique()))}

        # 创建反向映射
        self.standard_reverse_mapping = {v: k for k, v in self.standard_mapping.items()}
        self.level1_reverse_mapping = {v: k for k, v in self.level1_mapping.items()}
        self.level2_reverse_mapping = {v: k for k, v in self.level2_mapping.items()}
        self.level3_reverse_mapping = {v: k for k, v in self.level3_mapping.items()}

    def _save_mappings(self):
        """保存标签映射"""
        mappings = {
            'standard_mapping': self.standard_mapping,
            'level1_mapping': self.level1_mapping,
            'level2_mapping': self.level2_mapping,
            'level3_mapping': self.level3_mapping,
            'standard_reverse_mapping': self.standard_reverse_mapping,
            'level1_reverse_mapping': self.level1_reverse_mapping,
            'level2_reverse_mapping': self.level2_reverse_mapping,
            'level3_reverse_mapping': self.level3_reverse_mapping,
        }

        os.makedirs('./models', exist_ok=True)
        with open('./models/label_mappings.json', 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)

        logger.info("标签映射已保存")

    def _load_mappings(self):
        """加载标签映射"""
        mapping_path = './models/label_mappings.json'
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在: {mapping_path}")

        with open(mapping_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        self.standard_mapping = mappings['standard_mapping']
        self.level1_mapping = mappings['level1_mapping']
        self.level2_mapping = mappings['level2_mapping']
        self.level3_mapping = mappings['level3_mapping']

        self.standard_reverse_mapping = mappings['standard_reverse_mapping']
        self.level1_reverse_mapping = mappings['level1_reverse_mapping']
        self.level2_reverse_mapping = mappings['level2_reverse_mapping']
        self.level3_reverse_mapping = mappings['level3_reverse_mapping']

        logger.info("标签映射已加载")

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 使用jieba分词
        words = jieba.lcut(text)
        # 过滤掉单字符（除非是数字或字母）
        words = [word for word in words if len(word) > 1 or word.isdigit() or word.isalpha()]
        return ' '.join(words)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个样本"""
        row = self.data.iloc[idx]

        # 预处理文本
        product_name = str(row['product_name'])
        preprocessed_name = self._preprocess_text(product_name)

        # 分词和编码
        encoding = self.tokenizer(
            preprocessed_name,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )

        # 获取标签 - 兼容性处理未知标签
        try:
            labels_standard = torch.tensor(self.standard_mapping[row['standard_name']], dtype=torch.long)
        except KeyError:
            # 未知标签映射到特殊类别或忽略
            if not self.is_train:
                # 验证时跳过未知标签样本
                return None
            else:
                # 训练时不应该有未知标签
                raise KeyError(f"未知标准名称标签: {row['standard_name']}")

        try:
            labels_level1 = torch.tensor(self.level1_mapping[row['level1']], dtype=torch.long)
        except KeyError:
            if not self.is_train:
                return None
            else:
                raise KeyError(f"未知一级分类标签: {row['level1']}")

        try:
            labels_level2 = torch.tensor(self.level2_mapping[row['level2']], dtype=torch.long)
        except KeyError:
            if not self.is_train:
                return None
            else:
                raise KeyError(f"未知二级分类标签: {row['level2']}")

        try:
            labels_level3 = torch.tensor(self.level3_mapping[row['level3']], dtype=torch.long)
        except KeyError:
            if not self.is_train:
                return None
            else:
                raise KeyError(f"未知三级分类标签: {row['level3']}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_standard': labels_standard,
            'labels_level1': labels_level1,
            'labels_level2': labels_level2,
            'labels_level3': labels_level3,
            'product_name': product_name,
            'original_text': preprocessed_name,
        }


class DataCollator:
    """
    数据整理器，用于批处理
    """

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        整理批次数据 - 过滤无效样本
        """
        # 过滤掉None值（未知标签样本）
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch:
            # 如果整个批次都是无效的，返回空张量
            return {
                'input_ids': torch.empty(0, self.max_length, dtype=torch.long),
                'attention_mask': torch.empty(0, self.max_length, dtype=torch.long),
                'labels_standard': torch.empty(0, dtype=torch.long),
                'labels_level1': torch.empty(0, dtype=torch.long),
                'labels_level2': torch.empty(0, dtype=torch.long),
                'labels_level3': torch.empty(0, dtype=torch.long),
            }

        # 提取各个字段
        input_ids = [item['input_ids'] for item in valid_batch]
        attention_mask = [item['attention_mask'] for item in valid_batch]
        labels_standard = [item['labels_standard'] for item in valid_batch]
        labels_level1 = [item['labels_level1'] for item in valid_batch]
        labels_level2 = [item['labels_level2'] for item in valid_batch]
        labels_level3 = [item['labels_level3'] for item in valid_batch]

        # 堆叠张量
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels_standard': torch.stack(labels_standard),
            'labels_level1': torch.stack(labels_level1),
            'labels_level2': torch.stack(labels_level2),
            'labels_level3': torch.stack(labels_level3),
        }


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    tokenizer_name: str = "dienstag/chinese-bert-wwm-ext",
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    """
    # 初始化分词器
    tokenizer = load_tokenizer(tokenizer_name)

    # 创建数据集
    train_dataset = ProductDataset(train_path, tokenizer, max_length, is_train=True)
    val_dataset = ProductDataset(val_path, tokenizer, max_length, is_train=False)

    # 创建数据整理器
    data_collator = DataCollator(tokenizer, max_length)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    test_loader = None
    if test_path:
        test_dataset = ProductDataset(test_path, tokenizer, max_length, is_train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True,
        )

    # 保存分词器
    tokenizer.save_pretrained('./models/tokenizer')
    logger.info("分词器已保存到 ./models/tokenizer")

    return train_loader, val_loader, test_loader


def analyze_dataset(data_path: str):
    """
    分析数据集分布
    """
    try:
        encodings = ['utf-8', 'gbk', 'gb2312']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("无法读取文件")

        print("=== 数据集分析报告 ===")
        print(f"总样本数: {len(df)}")
        print(f"列名: {list(df.columns)}")

        # 各个分类的分布
        print("\n=== 标准名称分布 (Top 10) ===")
        standard_counts = df['standard_name'].value_counts()
        print(f"类别数: {len(standard_counts)}")
        print(standard_counts.head(10))

        print("\n=== 一级分类分布 ===")
        level1_counts = df['level1'].value_counts()
        print(f"类别数: {len(level1_counts)}")
        print(level1_counts)

        print("\n=== 二级分类分布 (Top 10) ===")
        level2_counts = df['level2'].value_counts()
        print(f"类别数: {len(level2_counts)}")
        print(level2_counts.head(10))

        print("\n=== 三级分类分布 (Top 10) ===")
        level3_counts = df['level3'].value_counts()
        print(f"类别数: {len(level3_counts)}")
        print(level3_counts.head(10))

        # 产品名称长度分布
        print("\n=== 产品名称长度分布 ===")
        df['name_length'] = df['product_name'].astype(str).str.len()
        print(f"平均长度: {df['name_length'].mean():.2f}")
        print(f"最短长度: {df['name_length'].min()}")
        print(f"最长长度: {df['name_length'].max()}")
        print(f"长度标准差: {df['name_length'].std():.2f}")

        # 缺失值统计
        print("\n=== 缺失值统计 ===")
        missing_counts = df.isnull().sum()
        print(missing_counts)

    except Exception as e:
        print(f"分析数据集失败: {e}")


if __name__ == "__main__":
    # 测试数据集
    print("测试数据集模块...")

    # 创建示例数据用于测试
    sample_data = {
        'product_name': ['苹果手机', '华为笔记本', '小米电视', '联想电脑', '三星平板'],
        'standard_name': ['手机', '笔记本电脑', '电视机', '笔记本电脑', '平板电脑'],
        'level1': ['电子产品', '电子产品', '电子产品', '电子产品', '电子产品'],
        'level2': ['手机数码', '电脑办公', '家用电器', '电脑办公', '手机数码'],
        'level3': ['智能手机', '笔记本电脑', '智能电视', '笔记本电脑', '平板电脑'],
    }

    # 创建示例CSV文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(f.name, index=False, encoding='utf-8')
        temp_path = f.name

    try:
        # 分析示例数据
        analyze_dataset(temp_path)

        # 测试数据集创建
        tokenizer = load_tokenizer("dienstag/chinese-bert-wwm-ext")
        dataset = ProductDataset(temp_path, tokenizer)

        print(f"\n数据集测试通过: {len(dataset)} 样本")

        # 测试单个样本
        sample = dataset[0]
        print(f"样本键: {sample.keys()}")
        print(f"input_ids形状: {sample['input_ids'].shape}")
        print(f"labels_standard: {sample['labels_standard']}")

    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        # 清理临时文件
        import os
        os.unlink(temp_path)