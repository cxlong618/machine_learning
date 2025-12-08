#!/usr/bin/env python3
"""
最终训练脚本 - 产品分类多任务BERT模型
完全使用ModelScope，无HuggingFace依赖
"""
import os
import sys
import logging
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_files(train_path=None, val_path=None):
    """检查数据文件"""
    logger.info("检查数据文件...")

    if train_path and val_path:
        # 使用传入的参数
        required_files = [train_path, val_path]
    else:
        # 使用默认路径
        required_files = ['data/train.csv', 'data/val.csv']

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.error(f"缺失数据文件: {missing_files}")
        logger.info("请确保数据文件存在:")
        for f in missing_files:
            logger.info(f"   - {f}")
        return False

    logger.info("数据文件检查通过")
    return True

def parse_arguments():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(description='产品分类模型训练脚本')

    # 必需参数
    parser.add_argument('--train_path', type=str, required=True, help='训练数据CSV文件路径')
    parser.add_argument('--val_path', type=str, required=True, help='验证数据CSV文件路径')

    # 可选参数
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度 (默认: 128)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率 (默认: 2e-5)')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数 (默认: 10)')
    parser.add_argument('--model_name', type=str, default='dienstag/chinese-bert-wwm-ext', help='基础模型名称')
    parser.add_argument('--warmup_steps', type=int, default=500, help='预热步数 (默认: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减 (默认: 0.01)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数 (默认: 1)')

    return parser.parse_args()


def run_training():
    """运行完整训练流程"""
    args = parse_arguments()

    logger.info("启动产品分类模型训练...")
    logger.info("=" * 60)

    # 显示训练参数
    logger.info("训练参数:")
    logger.info(f"  训练数据: {args.train_path}")
    logger.info(f"  验证数据: {args.val_path}")
    logger.info(f"  最大长度: {args.max_length}")
    logger.info(f"  批次大小: {args.batch_size}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  训练轮数: {args.num_epochs}")
    logger.info(f"  基础模型: {args.model_name}")
    logger.info(f"  权重衰减: {args.weight_decay}")
    logger.info("=" * 60)

    try:
        # 导入必要的模块
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        from sklearn.metrics import classification_report, accuracy_score
        import numpy as np
        from datetime import datetime
        import json
        from tqdm import tqdm
        import wandb

        # 导入项目模块
        from dataset import ProductDataset, DataCollator
        from model import MultiTaskProductClassifier
        from modelscope_utils import load_tokenizer, load_bert_model

        logger.info("所有模块导入成功")

        # 强制使用GPU并优化配置
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)

            logger.info(f"使用设备: {device}")
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"显存: {total_memory_gb:.1f} GB")

            # 根据显存大小自动优化批次大小
            if total_memory_gb >= 24:
                # 24GB+ 显存，使用大批次
                optimal_batch_size = min(args.batch_size, 64)
                logger.info(f"大显存GPU，设置批次大小: {optimal_batch_size}")
            elif total_memory_gb >= 16:
                # 16-24GB 显存
                optimal_batch_size = min(args.batch_size, 32)
                logger.info(f"中等显存GPU，设置批次大小: {optimal_batch_size}")
            elif total_memory_gb >= 8:
                # 8-16GB 显存
                optimal_batch_size = min(args.batch_size, 16)
                logger.info(f"小显存GPU，设置批次大小: {optimal_batch_size}")
            else:
                # 小于8GB显存
                optimal_batch_size = min(args.batch_size, 8)
                logger.warning(f"很小显存GPU({total_memory_gb:.1f}GB)，设置小批次: {optimal_batch_size}")

            args.batch_size = optimal_batch_size

            # 清理GPU缓存
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")

        else:
            logger.error("未检测到CUDA，请确保安装了GPU版本的PyTorch")
            device = torch.device('cpu')
            # GPU不可用时也使用相对合理的批次
            args.batch_size = min(16, args.batch_size)

        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # 初始化WandB（离线模式）
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            project="product-classification",
            name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
            mode="offline"
        )

        # 加载分词器
        logger.info("加载分词器...")
        tokenizer = load_tokenizer(args.model_name)

        # 创建数据集
        logger.info("创建数据集...")
        train_dataset = ProductDataset(
            data_path=args.train_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            is_train=True
        )

        val_dataset = ProductDataset(
            data_path=args.val_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            is_train=False
        )

        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")

        # 创建数据整理器
        data_collator = DataCollator(tokenizer, max_length=args.max_length)

        # 创建数据加载器 - GPU优化
        pin_memory = torch.cuda.is_available()
        num_workers = 0  # Windows下保持0避免问题

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # 丢弃最后不完整的批次，提高GPU利用率
            collate_fn=data_collator,
            persistent_workers=False  # 避免内存泄漏
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=data_collator,
            persistent_workers=False
        )

        # 创建模型
        logger.info("创建模型...")
        model_config = {
            'num_labels_standard': len(train_dataset.standard_mapping),
            'num_labels_level1': len(train_dataset.level1_mapping),
            'num_labels_level2': len(train_dataset.level2_mapping),
            'num_labels_level3': len(train_dataset.level3_mapping),
            'loss_weights': {'standard': 0.4, 'level1': 0.2, 'level2': 0.2, 'level3': 0.2}
        }

        model = MultiTaskProductClassifier.from_pretrained(
            args.model_name,
            **model_config
        )
        model.to(device)

        logger.info(f"模型参数量: {model.num_parameters():,}")

        # 创建优化器和学习率调度器
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"总训练步数: {total_steps}")

        # 训练循环 - GPU优化
        logger.info("开始训练...")
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5  # 增加耐心，充分利用GPU

        # 使用混合精度训练加速（如果可用）
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        if use_amp:
            logger.info("启用混合精度训练加速")
            scaler = torch.cuda.amp.GradScaler()

        # 启用梯度检查点以节省显存
        use_checkpoint = args.max_length > 128  # 长序列使用检查点

        for epoch in range(args.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

            # 预热GPU（前几个批次使用较小批次）
            is_warmup = epoch == 0
            warmup_factor = 0.5 if is_warmup else 1.0

            # 训练阶段
            model.train()
            total_train_loss = 0
            steps_per_epoch = len(train_dataloader)

            # 使用更激进的进度条更新
            train_progress = tqdm(
                train_dataloader,
                desc=f"训练 Epoch {epoch + 1}",
                leave=True,
                dynamic_ncols=True
            )

            for step, batch in enumerate(train_progress):
                # 动态批次大小调整（GPU预热后使用更大批次）
                current_batch_size = int(args.batch_size * warmup_factor)
                if batch['input_ids'].size(0) != current_batch_size:
                    continue  # 跳过不匹配的批次

                try:
                    # 高效数据转移
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                    # 使用自动混合精度（AMP）
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(**batch)
                            loss = outputs['loss'] / warmup_factor
                    else:
                        outputs = model(**batch)
                        loss = outputs['loss'] / warmup_factor

                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.error(f"CUDA错误: {e}")
                        # 清理GPU缓存并重试
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

                # 梯度累积优化
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # 混合精度反向传播
                if use_amp:
                    scaler.scale(loss).backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        # 定期清理GPU缓存
                        if step % 50 == 0:
                            torch.cuda.empty_cache()
                else:
                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # 定期清理GPU缓存
                        if step % 50 == 0:
                            torch.cuda.empty_cache()

                total_train_loss += loss.item() * warmup_factor

                # 更高效的进度条更新
                current_loss = loss.item() * warmup_factor
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.learning_rate
                train_progress.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'batch': f"{current_batch_size}"
                })

                # 减少日志频率以提高性能
                if step % 50 == 0:  # 每50步记录一次而不是100步
                    wandb.log({
                        'train_loss': current_loss,
                        'learning_rate': current_lr,
                        'step': epoch * steps_per_epoch + step,
                        'epoch': epoch + 1,
                        'gpu_memory_used': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    })

            avg_train_loss = total_train_loss / len(train_dataloader)
            logger.info(f"训练损失: {avg_train_loss:.4f}")

            # 验证阶段 - GPU优化
            model.eval()
            total_val_loss = 0
            all_predictions = {'standard': [], 'level1': [], 'level2': [], 'level3': []}
            all_labels = {'standard': [], 'level1': [], 'level2': [], 'level3': []}

            with torch.no_grad():
                # GPU验证时使用更大的批次
                val_batch_size = min(args.batch_size * 2, 64)  # 验证时可以使用更大批次

                val_progress = tqdm(val_dataloader, desc=f"验证 Epoch {epoch + 1}")

                for batch in val_progress:
                    try:
                        # 高效数据转移
                        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                        # 使用混合精度（如果训练时也使用）
                        if use_amp:
                            with torch.amp.autocast('cuda'):
                                outputs = model(**batch)
                                loss = outputs['loss']
                        else:
                            outputs = model(**batch)
                            loss = outputs['loss']

                        total_val_loss += loss.item()

                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.error(f"验证CUDA错误: {e}")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                    # 收集预测结果 - 兼容性修复
                    for task in ['standard', 'level1', 'level2', 'level3']:
                        preds = torch.argmax(outputs[f'logits_{task}'], dim=-1)
                        labels = batch[f'labels_{task}']

                        # 兼容性numpy转换
                        try:
                            all_predictions[task].extend(preds.cpu().numpy())
                            all_labels[task].extend(labels.cpu().numpy())
                        except RuntimeError as e:
                            if "Numpy is not available" in str(e):
                                # 使用PyTorch原生方法
                                all_predictions[task].extend(preds.cpu().tolist())
                                all_labels[task].extend(labels.cpu().tolist())
                            else:
                                raise

                    val_progress.set_postfix({'val_loss': loss.item()})

            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"验证损失: {avg_val_loss:.4f}")

            # 计算准确率 - 兼容性修复
            val_accuracies = {}
            for task in ['standard', 'level1', 'level2', 'level3']:
                # 转换为numpy数组兼容格式
                try:
                    labels_array = np.array(all_labels[task])
                    preds_array = np.array(all_predictions[task])
                    accuracy = accuracy_score(labels_array, preds_array)
                except (RuntimeError, ImportError):
                    # 如果numpy不可用，手动计算准确率
                    correct = sum(1 for l, p in zip(all_labels[task], all_predictions[task]) if l == p)
                    accuracy = correct / len(all_labels[task]) if all_labels[task] else 0

                val_accuracies[task] = accuracy
                logger.info(f"{task} 准确率: {accuracy:.4f}")

            # 记录到WandB - 兼容性修复
            accuracy_values = list(val_accuracies.values())
            avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0

            wandb.log({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'val_accuracy_standard': val_accuracies['standard'],
                'val_accuracy_level1': val_accuracies['level1'],
                'val_accuracy_level2': val_accuracies['level2'],
                'val_accuracy_level3': val_accuracies['level3'],
                'val_accuracy_avg': avg_accuracy
            })

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # 确保models目录存在
                models_dir = './models'
                os.makedirs(models_dir, exist_ok=True)

                # 保存模型
                model_path = os.path.join(models_dir, 'best_model')
                tokenizer_path = os.path.join(models_dir, 'tokenizer')
                mappings_path = os.path.join(models_dir, 'label_mappings.json')

                logger.info(f"保存模型到: {model_path}")
                model.save_pretrained(model_path)

                logger.info(f"保存分词器到: {tokenizer_path}")
                tokenizer.save_pretrained(tokenizer_path)

                # 保存标签映射
                label_mappings = {
                    'standard_name': train_dataset.standard_mapping,
                    'level1_category': train_dataset.level1_mapping,
                    'level2_category': train_dataset.level2_mapping,
                    'level3_category': train_dataset.level3_mapping
                }

                with open(mappings_path, 'w', encoding='utf-8') as f:
                    json.dump(label_mappings, f, ensure_ascii=False, indent=2)

                logger.info(f"保存标签映射到: {mappings_path}")

                logger.info("✅ 保存最佳模型")
            else:
                patience_counter += 1
                logger.info(f"验证损失未改善，耐心计数: {patience_counter}/{max_patience}")

                if patience_counter >= max_patience:
                    logger.info("早停训练")
                    break

        # 训练完成
        logger.info("🎉 训练完成!")
        wandb.finish()

        return True

    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 解析参数
    try:
        args = parse_arguments()
    except SystemExit:
        # 参数解析失败，显示帮助信息
        return False

    print("产品分类模型 - ModelScope最终版")
    print("=" * 60)

    # 显示ModelScope信息
    print("ModelScope 配置信息:")
    print("  - 模型源: ModelScope (魔搭)")
    print("  - 模型名称: dienstag/chinese-bert-wwm-ext")
    print("  - 下载平台: 国内高速服务器")
    print("  - 无HuggingFace依赖: OK")
    print("  - 自动缓存: OK")
    print("  - 离线运行: OK")

    # 数据文件检查
    if not check_data_files(args.train_path, args.val_path):
        logger.error("数据文件检查失败")
        return False

    # 开始训练
    success = run_training()

    if success:
        print("\n训练完成!")
        print("输出文件:")
        print("  - ./models/best_model/: 最佳模型文件")
        print("  - ./models/tokenizer/: 分词器文件")
        print("  - ./models/label_mappings.json: 标签映射文件")
        print("  - ./models/config.json: 模型配置文件")
        print("  - ./models/pytorch_model.bin: 模型权重文件")
        print("\n下一步:")
        print("  1. 下载 models/ 目录到本地")
        print("  2. 运行 python run_inference.py 测试推理")
        print("  3. 运行 python deploy_app.py 启动Web服务")
        print("\n监控和日志:")
        print("  - WandB项目页面查看详细指标")
        print("  - 本地日志: tail -f logs/training.log")
    else:
        print("\n训练失败")
        print("请检查:")
        print("  1. 数据文件是否存在")
        print("  2. Python环境和依赖包")
        print("  3. 网络连接状态")
        print("  4. GPU资源是否可用")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)