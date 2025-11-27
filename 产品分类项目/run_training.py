#!/usr/bin/env python3
"""
一键训练脚本 - 产品分类模型
适用于AutoDL GPU环境
"""
import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """检查环境配置"""
    logger.info("检查环境配置...")

    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA可用: {torch.cuda.get_device_name()}")
            logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.error("CUDA不可用")
            return False
    except ImportError:
        logger.error("PyTorch未安装")
        return False

    # 检查必要目录
    required_dirs = ['data', 'models', 'src', 'logs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"创建目录: {dir_name}")

    return True


def setup_environment():
    """设置环境变量"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_MODE'] = 'online'  # 可以改为 'offline' 如果不需要上传
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)

    logger.info("环境变量设置完成")


def run_training(args):
    """运行训练"""
    # 构建训练命令
    cmd = [
        sys.executable,
        'src/train.py',
        '--train_path', args.train_path,
        '--val_path', args.val_path,
        '--max_length', str(args.max_length),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--num_epochs', str(args.num_epochs),
        '--model_name', args.model_name
    ]

    if args.test_path:
        cmd.extend(['--test_path', args.test_path])

    logger.info("开始训练...")
    logger.info(f"训练命令: {' '.join(cmd)}")

    try:
        # 运行训练
        result = subprocess.run(cmd, check=True)
        logger.info("训练完成!")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='产品分类模型一键训练脚本')

    # 必需参数
    parser.add_argument('--train_path', type=str, required=True, help='训练数据CSV文件路径')
    parser.add_argument('--val_path', type=str, required=True, help='验证数据CSV文件路径')

    # 可选参数
    parser.add_argument('--test_path', type=str, help='测试数据CSV文件路径')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度 (默认: 128)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率 (默认: 2e-5)')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数 (默认: 10)')
    parser.add_argument('--model_name', type=str, default='dienstag/chinese-bert-wwm-ext',
                       help='基础模型 (默认: dienstag/chinese-bert-wwm-ext)')
    parser.add_argument('--skip_env_check', action='store_true', help='跳过环境检查')

    args = parser.parse_args()

    print("产品分类模型训练脚本")
    print("="*50)

    # 显示配置
    print("训练配置:")
    print(f"  训练数据: {args.train_path}")
    print(f"  验证数据: {args.val_path}")
    if args.test_path:
        print(f"  测试数据: {args.test_path}")
    print(f"  最大长度: {args.max_length}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  基础模型: {args.model_name}")
    print("="*50)

    # 检查环境
    if not args.skip_env_check:
        if not check_environment():
            logger.error("环境检查失败，使用 --skip_env_check 跳过检查")
            return 1

    # 设置环境
    setup_environment()

    # 检查数据文件
    for path, name in [(args.train_path, '训练数据'), (args.val_path, '验证数据')]:
        if not os.path.exists(path):
            logger.error(f"{name}文件不存在: {path}")
            return 1

    if args.test_path and not os.path.exists(args.test_path):
        logger.error(f"测试数据文件不存在: {args.test_path}")
        return 1

    # 运行训练
    success = run_training(args)

    if success:
        print("\n训练成功完成!")
        print("\n输出文件:")
        print("  - models/best_model.pt: 最佳模型文件")
        print("  - models/label_mappings.json: 标签映射文件")
        print("  - models/tokenizer/: 分词器文件")
        print("  - logs/training.log: 训练日志")
        print("\n下一步:")
        print("  1. 下载 models/ 目录到本地")
        print("  2. 运行 python run_inference.py 测试推理")
        print("  3. 运行 python deploy_app.py 启动Web服务")
        return 0
    else:
        print("\n训练失败!")
        print("请检查 logs/training.log 获取详细错误信息")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)