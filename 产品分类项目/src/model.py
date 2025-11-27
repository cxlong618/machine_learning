"""
多任务BERT模型架构
产品分类模型 - 同时预测4个分类任务
使用ModelScope完全集成
"""
import torch
import torch.nn as nn
from transformers import BertConfig, BertPreTrainedModel
from modelscope_utils import load_tokenizer, load_bert_model, ModelScopeLoader
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    snapshot_download = None
    print("警告: ModelScope未安装，将使用备用方案")
from typing import Dict, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskProductClassifier(BertPreTrainedModel):
    """
    基于BERT的多任务产品分类模型
    完全基于ModelScope，HuggingFace作为备用

    输入：产品名称文本
    输出：
        - 标准名称分类 (num_classes: 936)
        - 一级分类 (num_classes: 24)
        - 二级分类 (num_classes: 78)
        - 三级分类 (num_classes: 138)
    """

    def __init__(self, config):
        super().__init__(config)

        # 获取分类数量
        self.num_labels_standard = config.num_labels_standard
        self.num_labels_level1 = config.num_labels_level1
        self.num_labels_level2 = config.num_labels_level2
        self.num_labels_level3 = config.num_labels_level3

        # 损失权重（用于平衡不同任务的损失）
        self.loss_weights = config.loss_weights

        # BERT主干网络
        self.bert = None  # 将在 _init_bert_weights 中加载

        # 分类器头（将在BERT加载后初始化）
        self.classifier_standard = None
        self.classifier_level1 = None
        self.classifier_level2 = None
        self.classifier_level3 = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        从预训练模型创建分类器实例
        完全使用ModelScope，不依赖HuggingFace
        """
        # 从参数中提取分类数量
        num_labels_standard = kwargs.pop('num_labels_standard', 936)
        num_labels_level1 = kwargs.pop('num_labels_level1', 24)
        num_labels_level2 = kwargs.pop('num_labels_level2', 78)
        num_labels_level3 = kwargs.pop('num_labels_level3', 138)
        loss_weights = kwargs.pop('loss_weights', {'standard': 0.4, 'level1': 0.2, 'level2': 0.2, 'level3': 0.2})

        # 创建基础配置
        from modelscope_utils import load_model_config
        config = load_model_config(model_name_or_path)

        # 添加分类数量到配置
        config.num_labels_standard = num_labels_standard
        config.num_labels_level1 = num_labels_level1
        config.num_labels_level2 = num_labels_level2
        config.num_labels_level3 = num_labels_level3
        config.loss_weights = loss_weights
        config.hidden_dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)

        # 创建模型实例
        model = cls(config)

        # 初始化BERT权重
        model._init_bert_weights(model_name_or_path)

        return model

    def _init_bert_weights(self, model_name_or_path):
        """
        初始化BERT权重
        优先使用ModelScope下载，HuggingFace作为备用
        """
        # 检查是否是本地路径
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            logger.info(f"检测到本地模型目录: {model_name_or_path}")
            # 从本地目录加载
            config_path = os.path.join(model_name_or_path, "config.json")
            model_path = os.path.join(model_name_or_path, "pytorch_model.bin")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"找不到配置文件: {config_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到模型文件: {model_path}")

            # 加载配置
            from transformers import BertConfig
            config = BertConfig.from_pretrained(model_name_or_path, local_files_only=True)

            # 创建模型实例
            from transformers import BertModel
            bert = BertModel(config)

            # 加载权重
            import torch
            state_dict = torch.load(model_path, map_location='cpu')

            # 如果是完整模型，提取state_dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # 加载权重（允许部分匹配）
            bert.load_state_dict(state_dict, strict=False)

            self.bert = bert
            logger.info(f"成功从本地加载模型: {model_name_or_path}")

            # 初始化分类器头
            self._init_classifiers()
            return

        # 如果不是本地路径，尝试从ModelScope加载
        else:
            logger.info("尝试从ModelScope加载模型...")
            if MODELSCOPE_AVAILABLE and snapshot_download is not None:
                try:
                    # 从ModelScope下载模型文件
                    model_dir = snapshot_download(model_name_or_path)
                    logger.info(f"模型文件下载完成: {model_dir}")

                    # 加载模型
                    self.bert = load_bert_model(model_name_or_path)
                    logger.info(f"成功从ModelScope加载模型: {model_name_or_path}")
                except Exception as e:
                    logger.error(f"ModelScope模型加载失败: {e}")
                    # Fallback到HuggingFace
                    logger.info("fallback到HuggingFace...")
                    try:
                        from transformers import BertModel, BertConfig
                        config = BertConfig.from_pretrained(model_name_or_path)
                        bert = BertModel.from_pretrained(model_name_or_path)
                        self.bert = bert
                        logger.info(f"成功从HuggingFace加载模型: {model_name_or_path}")
                    except Exception as he_error:
                        logger.error(f"HuggingFace加载也失败: {he_error}")
                        raise RuntimeError(f"无法从任何来源加载模型: {model_name_or_path}")
            else:
                # 直接使用HuggingFace
                logger.info("ModelScope不可用，直接使用HuggingFace...")
                try:
                    from transformers import BertModel, BertConfig
                    config = BertConfig.from_pretrained(model_name_or_path)
                    bert = BertModel.from_pretrained(model_name_or_path)
                    self.bert = bert
                    logger.info(f"成功从HuggingFace加载模型: {model_name_or_path}")
                except Exception as he_error:
                    logger.error(f"HuggingFace加载失败: {he_error}")
                    raise RuntimeError(f"无法从HuggingFace加载模型: {model_name_or_path}")

        # 初始化分类器头
        self._init_classifiers()

    def _init_classifiers(self):
        """初始化分类器头"""
        hidden_size = self.bert.config.hidden_size

        # 初始化多任务分类器
        self.classifier_standard = nn.Linear(hidden_size, self.num_labels_standard)
        self.classifier_level1 = nn.Linear(hidden_size, self.num_labels_level1)
        self.classifier_level2 = nn.Linear(hidden_size, self.num_labels_level2)
        self.classifier_level3 = nn.Linear(hidden_size, self.num_labels_level3)

        # 初始化权重
        for classifier in [self.classifier_standard, self.classifier_level1,
                          self.classifier_level2, self.classifier_level3]:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.zeros_(classifier.bias)

        logger.info("分类器头初始化完成")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                 labels_standard=None, labels_level1=None, labels_level2=None, labels_level3=None):
        """
        前向传播
        """
        outputs = {}

        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # 获取序列表示（[CLS]标记）
        pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_size]

        # 多任务分类
        logits_standard = self.classifier_standard(pooled_output)
        logits_level1 = self.classifier_level1(pooled_output)
        logits_level2 = self.classifier_level2(pooled_output)
        logits_level3 = self.classifier_level3(pooled_output)

        # 计算损失（用于监控）
        loss = torch.tensor(0.0, device=input_ids.device)

        if labels_standard is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_standard = loss_fct(logits_standard, labels_standard)
            loss += self.loss_weights['standard'] * loss_standard

        if labels_level1 is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_level1 = loss_fct(logits_level1, labels_level1)
            loss += self.loss_weights['level1'] * loss_level1

        if labels_level2 is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_level2 = loss_fct(logits_level2, labels_level2)
            loss += self.loss_weights['level2'] * loss_level2

        if labels_level3 is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_level3 = loss_fct(logits_level3, labels_level3)
            loss += self.loss_weights['level3'] * loss_level3

        outputs['loss'] = loss
        outputs['logits_standard'] = logits_standard
        outputs['logits_level1'] = logits_level1
        outputs['logits_level2'] = logits_level2
        outputs['logits_level3'] = logits_level3

        return outputs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        仅用于推理，返回预测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # 获取预测结果
            standard_pred = torch.argmax(outputs['logits_standard'], dim=-1)
            level1_pred = torch.argmax(outputs['logits_level1'], dim=-1)
            level2_pred = torch.argmax(outputs['logits_level2'], dim=-1)
            level3_pred = torch.argmax(outputs['logits_level3'], dim=-1)

            return standard_pred, level1_pred, level2_pred, level3_pred

    def predict_with_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        推理并返回概率
        """
        self.eval()
        with torch.no_grad():
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # 获取概率
            probs_standard = torch.softmax(outputs['logits_standard'], dim=-1)
            probs_level1 = torch.softmax(outputs['logits_level1'], dim=-1)
            probs_level2 = torch.softmax(outputs['logits_level2'], dim=-1)
            probs_level3 = torch.softmax(outputs['logits_level3'], dim=-1)

            # 获取预测
            standard_pred = torch.argmax(probs_standard, dim=-1)
            level1_pred = torch.argmax(probs_level1, dim=-1)
            level2_pred = torch.argmax(probs_level2, dim=-1)
            level3_pred = torch.argmax(probs_level3, dim=-1)

            return {
                'standard_pred': standard_pred.item(),
                'level1_pred': level1_pred.item(),
                'level2_pred': level2_pred.item(),
                'level3_pred': level3_pred.item(),
                'confidence_standard': probs_standard[torch.arange(len(probs_standard))].item(),
                'confidence_level1': probs_level1[torch.arange(len(probs_level1))].item(),
                'confidence_level2': probs_level2[torch.arange(len(probs_level1))].item(),
                'confidence_level3': probs_level3[torch.arange(len(probs_level1))].item()
            }


# 为了保持向后兼容，创建工厂函数
def create_model_loader(config):
    """
    根据配置创建模型加载器
    """
    logger.info("创建ModelScope模型加载器")
    return ModelScopeLoader(model_source="modelscope")


# 默认加载器实例
default_loader = ModelScopeLoader(model_source="modelscope")

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
    """
    try:
        logger.info(f"从ModelScope下载模型文件: {model_name}")
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        logger.info(f"模型文件下载完成: {model_dir}")
        return model_dir
    except Exception as e:
        logger.error(f"模型文件下载失败: {e}")
        raise
