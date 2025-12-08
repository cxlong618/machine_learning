"""
推理模块 - 产品分类模型推理接口
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer
from modelscope_utils import load_tokenizer
import json
import logging
from typing import Dict, Tuple, Optional, List
import os
from pathlib import Path
import numpy as np
import time

from model import MultiTaskProductClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductInference:
    """产品分类推理器"""

    def __init__(self, model_path: str = "./models/best_model"):
        """
        初始化推理器

        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型和分词器
        self.model = None
        self.tokenizer = None
        self.label_mappings = None

        self._load_model()
        self._load_tokenizer()
        self._load_label_mappings()

        logger.info(f"推理器初始化完成，使用设备: {self.device}")

    def _load_model(self):
        """加载训练好的模型"""
        try:
            # 检查是否是目录格式（新保存格式）
            if os.path.isdir(self.model_path):
                logger.info(f"检测到目录格式模型: {self.model_path}")

                # 使用新的from_saved_model方法加载
                self.model = MultiTaskProductClassifier.from_saved_model(
                    self.model_path)

            else:
                # 兼容旧的单文件格式
                logger.info(f"检测到单文件格式模型: {self.model_path}")

                # 加载检查点
                checkpoint = torch.load(
                    self.model_path, map_location=self.device)

                # 获取模型配置
                if 'config' in checkpoint:
                    # 如果配置已保存
                    from model import ProductClassifierConfig
                    config = ProductClassifierConfig()
                    config.__dict__.update(checkpoint['config'])
                else:
                    # 使用默认配置
                    from model import ProductClassifierConfig
                    config = ProductClassifierConfig()

                # 创建模型（直接实例化，避免from_pretrained）
                from transformers import BertConfig

                # 创建配置
                bert_config = BertConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.hidden_dropout_prob,
                    max_position_embeddings=config.max_length +
                    2,  # +2 for [CLS] and [SEP]
                    vocab_size=21128,  # 中文BERT词汇表大小
                )

                # 添加自定义配置
                bert_config.num_labels_standard = config.num_labels_standard
                bert_config.num_labels_level1 = config.num_labels_level1
                bert_config.num_labels_level2 = config.num_labels_level2
                bert_config.num_labels_level3 = config.num_labels_level3
                bert_config.loss_weights = config.loss_weights

                # 创建模型实例
                self.model = MultiTaskProductClassifier(bert_config)

                # 加载权重
                if torch.cuda.device_count() > 1:
                    # 处理多GPU保存的模型
                    state_dict = {}
                    for k, v in checkpoint['model_state_dict'].items():
                        name = k.replace('module.', '') if k.startswith(
                            'module.') else k
                        state_dict[name] = v
                else:
                    state_dict = checkpoint['model_state_dict']

                self.model.load_state_dict(state_dict)

            # 确保模型在正确的设备上
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"模型加载成功，来自: {self.model_path}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _load_tokenizer(self):
        """加载分词器"""
        try:
            tokenizer_path = "./models/tokenizer"
            if os.path.exists(tokenizer_path):
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            else:
                # 如果本地没有，使用预训练模型
                self.tokenizer = load_tokenizer(
                    "dienstag/chinese-bert-wwm-ext")
                logger.warning("未找到本地分词器，使用预训练模型")

        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            raise

    def _load_label_mappings(self):
        """加载标签映射"""
        try:
            mapping_path = "./models/label_mappings.json"
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"标签映射文件不存在: {mapping_path}")

            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.label_mappings = json.load(f)

            # 创建反向映射以提高查找效率
            self.reverse_mappings = {}
            task_mappings = {
                'standard': 'standard_name',
                'level1': 'level1_category',
                'level2': 'level2_category',
                'level3': 'level3_category'
            }

            for label_type, mapping_key in task_mappings.items():
                if mapping_key in self.label_mappings:
                    reverse_key = f"{label_type}_reverse_mapping"
                    self.reverse_mappings[reverse_key] = {str(idx): name for name, idx in self.label_mappings[mapping_key].items()}
                    logger.info(f"创建反向映射: {reverse_key}, 包含 {len(self.reverse_mappings[reverse_key])} 个标签")
                else:
                    logger.warning(f"未找到映射键: {mapping_key}")

            logger.info("标签映射加载成功")

        except Exception as e:
            logger.error(f"标签映射加载失败: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        try:
            import jieba
            # 使用jieba分词
            words = jieba.lcut(str(text))
            # 过滤掉单字符（除非是数字或字母）
            words = [word for word in words if len(
                word) > 1 or word.isdigit() or word.isalpha()]
            return ' '.join(words)
        except ImportError:
            logger.warning("jieba未安装，使用原始文本")
            return str(text)

    def predict(self, product_name: str, return_prob: bool = False) -> Dict:
        """
        预测产品分类

        Args:
            product_name: 产品名称
            return_prob: 是否返回概率

        Returns:
            预测结果字典
        """
        start_time = time.time()

        try:
            # 预处理文本
            preprocessed_name = self._preprocess_text(product_name)

            # 分词和编码
            encoding = self.tokenizer(
                preprocessed_name,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt',
            )

            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # 预测
            with torch.no_grad():
                if return_prob:
                    # 带置信度的预测
                    result = self.model.predict_with_prob(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                else:
                    # 简单预测
                    result = self.model.predict(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

            # 解码结果
            if return_prob:
                predictions = {
                    'standard_name': self._decode_label(result['standard'][0], 'standard'),
                    'level1_category': self._decode_label(result['level1'][0], 'level1'),
                    'level2_category': self._decode_label(result['level2'][0], 'level2'),
                    'level3_category': self._decode_label(result['level3'][0], 'level3'),
                    'confidence_standard': float(result['standard'][1]),
                    'confidence_level1': float(result['level1'][1]),
                    'confidence_level2': float(result['level2'][1]),
                    'confidence_level3': float(result['level3'][1]),
                }

                # 综合置信度
                predictions['overall_confidence'] = (
                    predictions['confidence_standard'] * 0.4 +
                    predictions['confidence_level1'] * 0.2 +
                    predictions['confidence_level2'] * 0.2 +
                    predictions['confidence_level3'] * 0.2
                )
            else:
                predictions = {
                    'standard_name': self._decode_label(result[0], 'standard'),
                    'level1_category': self._decode_label(result[1], 'level1'),
                    'level2_category': self._decode_label(result[2], 'level2'),
                    'level3_category': self._decode_label(result[3], 'level3'),
                }

            # 添加元信息
            predictions['input_text'] = product_name
            predictions['processed_text'] = preprocessed_name
            predictions['response_time'] = f"{(time.time() - start_time)*1000:.2f}ms"

            return predictions

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'error': str(e),
                'input_text': product_name,
                'response_time': f"{(time.time() - start_time)*1000:.2f}ms"
            }

    def _decode_label(self, label_idx: int, label_type: str) -> str:
        """解码标签"""
        try:
            # 确保label_idx是整数
            label_idx_int = int(label_idx)

            # 使用预创建的反向映射进行快速查找
            reverse_key = f"{label_type}_reverse_mapping"
            if reverse_key in self.reverse_mappings:
                result = self.reverse_mappings[reverse_key].get(str(label_idx_int))
                if result:
                    return result
                else:
                    logger.warning(f"未找到标签索引: {label_idx_int}, 类型: {label_type}")
                    return f"未知标签_{label_idx_int}"
            else:
                logger.warning(f"未找到反向映射类型: {label_type}")
                return f"未知标签类型: {label_type}"

        except Exception as e:
            logger.error(f"标签解码错误: {e}, label_idx={label_idx}, label_type={label_type}")
            return f"解码错误_{label_idx}"

    def predict_batch(self, product_names: List[str], return_prob: bool = False) -> List[Dict]:
        """
        批量预测

        Args:
            product_names: 产品名称列表
            return_prob: 是否返回概率

        Returns:
            预测结果列表
        """
        results = []
        for product_name in product_names:
            result = self.predict(product_name, return_prob)
            results.append(result)

        return results

    def get_top_k_predictions(self, product_name: str, k: int = 5) -> Dict:
        """
        获取Top-K预测结果

        Args:
            product_name: 产品名称
            k: 返回的top数量

        Returns:
            包含top-k结果的字典
        """
        try:
            # 预处理文本
            preprocessed_name = self._preprocess_text(product_name)

            # 分词和编码
            encoding = self.tokenizer(
                preprocessed_name,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt',
            )

            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # 获取预测概率
            with torch.no_grad():
                result = self.model.predict_with_prob(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # 获取Top-K结果
            top_k_results = {}
            for task in ['standard', 'level1', 'level2', 'level3']:
                if task in result['probs']:
                    probs = result['probs'][task].cpu().numpy()[0]
                    top_k_indices = np.argsort(probs)[-k:][::-1]

                    top_k_results[task] = []
                    for idx in top_k_indices:
                        label = self._decode_label(idx, task)
                        prob = float(probs[idx])
                        top_k_results[task].append({
                            'label': label,
                            'probability': prob
                        })

            # 添加基础信息
            top_k_results['input_text'] = product_name
            top_k_results['processed_text'] = preprocessed_name

            return top_k_results

        except Exception as e:
            logger.error(f"Top-K预测失败: {e}")
            return {'error': str(e)}

    def evaluate_performance(self, num_samples: int = 100) -> Dict:
        """
        评估推理性能

        Args:
            num_samples: 测试样本数

        Returns:
            性能统计结果
        """
        import random

        # 生成随机测试样本
        test_products = [
            "苹果iPhone手机", "华为笔记本电脑", "小米电视", "联想电脑", "三星平板",
            "戴尔服务器", "索尼相机", "佳能打印机", "惠普扫描仪", "路由器",
            "交换机", "投影仪", "音响设备", "耳机", "键盘", "鼠标"
        ]

        # 如果样本不够，重复使用
        while len(test_products) < num_samples:
            test_products.extend(test_products)

        test_products = test_products[:num_samples]

        # 测试推理速度
        times = []
        for product_name in test_products:
            start_time = time.time()
            _ = self.predict(product_name)
            times.append((time.time() - start_time) * 1000)  # 转换为毫秒

        # 计算统计信息
        times = np.array(times)
        performance_stats = {
            'num_samples': num_samples,
            'avg_time_ms': float(np.mean(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'std_time_ms': float(np.std(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'throughput_qps': num_samples / (np.sum(times) / 1000),  # 每秒查询数
        }

        logger.info("性能评估完成:")
        for key, value in performance_stats.items():
            logger.info(f"  {key}: {value}")

        return performance_stats


# 全局推理器实例
_inference_instance = None


def get_inference_instance(model_path: str = "./models/best_model") -> ProductInference:
    """获取推理器实例（单例模式）"""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = ProductInference(model_path)
    return _inference_instance


if __name__ == "__main__":
    # 测试推理器
    print("🧪 测试产品分类推理器...")

    try:
        # 创建推理器
        inference = ProductInference()

        # 测试单个预测
        test_products = [
            "手术无影灯",
            "彩色超声",
            "4K腹腔镜",
            "CT",
            "胃镜"
        ]

        print("\n🔍 单个预测测试:")
        for product in test_products:
            print(f"\n产品: {product}")
            result = inference.predict(product, return_prob=True)
            print(
                f"  标准名称: {result['standard_name']} (置信度: {result.get('confidence_standard', 'N/A'):.3f})")
            print(
                f"  一级分类: {result['level1_category']} (置信度: {result.get('confidence_level1', 'N/A'):.3f})")
            print(
                f"  二级分类: {result['level2_category']} (置信度: {result.get('confidence_level2', 'N/A'):.3f})")
            print(
                f"  三级分类: {result['level3_category']} (置信度: {result.get('confidence_level3', 'N/A'):.3f})")
            print(f"  综合置信度: {result.get('overall_confidence', 'N/A'):.3f}")
            print(f"  响应时间: {result['response_time']}")

        # 测试批量预测
        print("\n📊 批量预测测试:")
        batch_results = inference.predict_batch(
            test_products[:3], return_prob=True)
        for i, result in enumerate(batch_results):
            print(
                f"  产品{i+1}: {result['standard_name']} - {result.get('overall_confidence', 'N/A'):.3f}")

        # 测试Top-K预测
        print("\n🎯 Top-K预测测试:")
        top_k_result = inference.get_top_k_predictions(test_products[0], k=3)
        print(f"产品: {top_k_result['input_text']}")
        print("Top-3 标准名称:")
        for i, item in enumerate(top_k_result['standard']):
            print(f"  {i+1}. {item['label']} - {item['probability']:.4f}")

        # 性能测试
        print("\n⚡ 性能测试:")
        performance = inference.evaluate_performance(num_samples=50)
        print(f"  平均响应时间: {performance['avg_time_ms']:.2f}ms")
        print(f"  吞吐量: {performance['throughput_qps']:.2f} QPS")

        print("\n✅ 推理器测试通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请确保:")
        print("  1. 模型文件已生成 (./models/best_model/)")
        print("  2. 标签映射文件已生成 (./models/label_mappings.json)")
        print("  3. 分词器文件已保存 (./models/tokenizer/)")
