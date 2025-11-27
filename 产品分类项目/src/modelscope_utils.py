"""
ModelScope模型下载和加载工具
完全基于ModelScope，不依赖HuggingFace
"""
import os
import logging
from typing import Optional, Dict
import torch
import torch.nn as nn
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("警告: ModelScope未安装，将使用备用方案")

try:
    from transformers import BertConfig, BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: Transformers未安装")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelScopeLoader:
    """ModelScope模型加载器 - 完全不依赖HuggingFace"""

    def __init__(self, model_source: str = "modelscope"):
        """
        初始化加载器

        Args:
            model_source: 固定为 'modelscope'
        """
        self.model_source = model_source
        if model_source != "modelscope":
            raise ValueError("当前版本仅支持ModelScope，不支持其他模型源")

    def load_tokenizer(self, model_name: str, cache_dir: Optional[str] = None):
        """
        从ModelScope加载分词器

        Args:
            model_name: 模型名称
            cache_dir: 缓存目录

        Returns:
            分词器实例
        """
        try:
            logger.info(f"从ModelScope下载分词器: {model_name}")
            model_dir = snapshot_download(model_name, cache_dir=cache_dir)

            # 手动加载分词器文件
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
            vocab_path = os.path.join(model_dir, "vocab.txt")

            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"找不到词汇表文件: {vocab_path}")

            # 使用transformers的BertTokenizer，但从本地文件加载
            from transformers import BertTokenizer
            tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

            logger.info(f"分词器加载成功，缓存位置: {model_dir}")
            return tokenizer

        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            raise

    def load_bert_model(self, model_name: str, cache_dir: Optional[str] = None):
        """
        从ModelScope加载BERT模型

        Args:
            model_name: 模型名称
            cache_dir: 缓存目录

        Returns:
            BertModel实例
        """
        try:
            logger.info(f"从ModelScope下载BERT模型: {model_name}")
            model_dir = snapshot_download(model_name, cache_dir=cache_dir)

            # 手动加载模型权重
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            config_path = os.path.join(model_dir, "config.json")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到模型权重文件: {model_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"找不到配置文件: {config_path}")

            # 加载配置
            config = BertConfig.from_pretrained(model_dir, local_files_only=True)

            # 创建模型并加载权重
            from transformers import BertModel
            model = BertModel(config)
            state_dict = torch.load(model_path, map_location='cpu')

            # 如果是完整模型，提取state_dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()

            model.load_state_dict(state_dict, strict=False)  # 使用strict=False以允许部分权重加载

            logger.info(f"BERT模型加载成功，缓存位置: {model_dir}")
            return model

        except Exception as e:
            logger.error(f"BERT模型加载失败: {e}")
            raise

    def load_model_config(self, model_name: str, cache_dir: Optional[str] = None) -> BertConfig:
        """
        从ModelScope加载模型配置

        Args:
            model_name: 模型名称
            cache_dir: 缓存目录

        Returns:
            BertConfig实例
        """
        try:
            logger.info(f"从ModelScope下载配置: {model_name}")
            model_dir = snapshot_download(model_name, cache_dir=cache_dir)

            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"找不到配置文件: {config_path}")

            # 加载配置
            config = BertConfig.from_pretrained(model_dir, local_files_only=True)
            logger.info(f"配置加载成功，缓存位置: {model_dir}")
            return config

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

def create_model_loader(config) -> ModelScopeLoader:
    """
    根据配置创建模型加载器

    Args:
        config: 配置对象或字典

    Returns:
        ModelScopeLoader实例
    """
    # 获取模型源配置，强制使用ModelScope
    logger.info("创建ModelScope模型加载器")
    return ModelScopeLoader(model_source="modelscope")

# 默认加载器实例
default_loader = ModelScopeLoader()

def load_tokenizer(model_name: str, **kwargs):
    """便捷函数：加载分词器"""
    return default_loader.load_tokenizer(model_name, **kwargs)

def load_bert_model(model_name: str, **kwargs):
    """便捷函数：加载BERT模型"""
    return default_loader.load_bert_model(model_name, **kwargs)

def load_model_config(model_name: str, **kwargs):
    """便捷函数：加载模型配置"""
    return default_loader.load_model_config(model_name, **kwargs)

def download_model_files(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    下载模型文件到本地

    Args:
        model_name: 模型名称
        cache_dir: 缓存目录

    Returns:
        模型文件目录路径
    """
    try:
        logger.info(f"下载ModelScope模型文件: {model_name}")
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        logger.info(f"模型文件下载完成: {model_dir}")
        return model_dir
    except Exception as e:
        logger.error(f"模型文件下载失败: {e}")
        raise

def load_simple_bert_model(
    model_name: str,
    num_labels_standard: int = 936,
    num_labels_level1: int = 24,
    num_labels_level2: int = 78,
    num_labels_level3: int = 138,
    loss_weights: Optional[Dict] = None
) -> BertModel:
    """
    简单加载BERT模型

    Args:
        model_name: 模型名称
        num_labels_standard: 标准名称分类数量
        num_labels_level1: 一级分类数量
        num_labels_level2: 二级分类数量
        num_labels_level3: 三级分类数量
        loss_weights: 损失权重

    Returns:
        BertModel实例
    """
    return default_loader.load_bert_model(model_name)