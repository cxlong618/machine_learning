#!/usr/bin/env python3
"""
训练优化配置器 - 根据硬件自动优化训练参数
"""
import sys
import json
import os

def detect_hardware():
    """检测硬件配置"""
    try:
        import torch
        hardware_info = {
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': os.cpu_count(),
        }

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            hardware_info['gpu_count'] = gpu_count

            # 获取GPU信息
            gpus = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_gb': props.total_memory / 1e9,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multiprocessor_count
                }
                gpus.append(gpu_info)

            hardware_info['gpus'] = gpus
            hardware_info['total_gpu_memory'] = sum(gpu['memory_gb'] for gpu in gpus)

        return hardware_info
    except ImportError:
        return {'cuda_available': False, 'cpu_count': os.cpu_count()}

def recommend_training_config(hardware_info):
    """根据硬件推荐训练配置"""
    config = {}

    if not hardware_info.get('cuda_available', False):
        # CPU配置
        config = {
            'script': 'cpu_train.py',
            'batch_size': 8,
            'learning_rate': '2e-5',
            'num_epochs': 3,
            'max_length': 64,
            'description': 'CPU训练配置 - 小批次大小和短序列',
            'training_time_hours': 6,  # 估计时间
        }
    else:
        # GPU配置
        total_memory = hardware_info.get('total_gpu_memory', 8)
        gpu_count = hardware_info.get('gpu_count', 1)

        if total_memory >= 24:
            # 高端GPU配置
            config = {
                'script': 'gpu_train.py',
                'batch_size': 64,
                'learning_rate': '3e-5',
                'num_epochs': 5,
                'max_length': 128,
                'description': '高端GPU配置 (≥24GB) - 大批次快速训练',
                'training_time_hours': 1,
                'estimated_speedup': '10-20x CPU'
            }
        elif total_memory >= 16:
            # 中高端GPU配置
            config = {
                'script': 'gpu_train.py',
                'batch_size': 48,
                'learning_rate': '3e-5',
                'num_epochs': 5,
                'max_length': 128,
                'description': '中高端GPU配置 (16-24GB) - 中大批次',
                'training_time_hours': 2,
                'estimated_speedup': '8-15x CPU'
            }
        elif total_memory >= 12:
            # 中端GPU配置
            config = {
                'script': 'gpu_train.py',
                'batch_size': 32,
                'learning_rate': '2.5e-5',
                'num_epochs': 5,
                'max_length': 128,
                'description': '中端GPU配置 (12-16GB) - 平衡批次大小',
                'training_time_hours': 3,
                'estimated_speedup': '6-12x CPU'
            }
        elif total_memory >= 8:
            # 入门级GPU配置
            config = {
                'script': 'src/train.py',  # 使用原始脚本但优化
                'batch_size': 16,
                'learning_rate': '2e-5',
                'num_epochs': 8,
                'max_length': 96,
                'description': '入门级GPU配置 (8-12GB) - 中等批次大小',
                'training_time_hours': 4,
                'estimated_speedup': '5-8x CPU'
            }
        else:
            # 低端GPU配置
            config = {
                'script': 'src/train.py',
                'batch_size': 8,
                'learning_rate': '2e-5',
                'num_epochs': 10,
                'max_length': 64,
                'description': '低端GPU配置 (<8GB) - 小批次大小',
                'training_time_hours': 6,
                'estimated_speedup': '3-5x CPU'
            }

    return config

def generate_training_command(config, train_path='data/train.csv', val_path='data/val.csv'):
    """生成训练命令"""
    script = config['script']

    base_cmd = f"python {script}"

    if script == 'gpu_train.py' or script == 'cpu_train.py':
        # 优化脚本不需要额外参数
        cmd = base_cmd
    else:
        # 原始脚本需要完整参数
        params = [
            f"--train_path {train_path}",
            f"--val_path {val_path}",
            f"--batch_size {config['batch_size']}",
            f"--learning_rate {config['learning_rate']}",
            f"--num_epochs {config['num_epochs']}",
            f"--max_length {config['max_length']}"
        ]
        cmd = f"{base_cmd} {' '.join(params)}"

    return cmd

def main():
    print("🖥️  硬件检测与训练优化")
    print("=" * 60)

    # 检测硬件
    print("检测硬件配置...")
    hardware = detect_hardware()

    print(f"CPU核心数: {hardware['cpu_count']}")
    print(f"CUDA可用: {'是' if hardware['cuda_available'] else '否'}")

    if hardware.get('cuda_available'):
        print(f"GPU数量: {hardware['gpu_count']}")
        print(f"GPU信息:")
        for i, gpu in enumerate(hardware.get('gpus', [])):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
            print(f"      计算能力: {gpu['compute_capability']}")

    print()

    # 推荐配置
    print("推荐训练配置:")
    print("-" * 40)
    config = recommend_training_config(hardware)

    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")

    print(f"\n📝 配置说明:")
    print(f"  {config['description']}")
    print(f"  ⏱️  预计训练时间: {config.get('training_time_hours', 'N/A')}小时")
    if 'estimated_speedup' in config:
        print(f"  🚀 相比CPU速度提升: {config['estimated_speedup']}")

    # 生成训练命令
    cmd = generate_training_command(config)
    print(f"\n🚀 推荐训练命令:")
    print(f"  {cmd}")

    # 保存配置
    config_file = 'training_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'hardware': hardware,
            'recommended_config': config,
            'command': cmd
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 配置已保存到: {config_file}")

    print("\n🎯 快速开始:")
    print("1. 运行上面的推荐命令")
    print("2. 或者运行优化后的脚本:")

    if hardware.get('cuda_available'):
        print("   python gpu_train.py  # GPU优化版本")
    else:
        print("   python cpu_train.py  # CPU稳定版本")

    print("3. 或者手动训练:")
    print("   python src/train.py --train_path data/train.csv --val_path data/val.csv")

    print("\n" + "=" * 60)
    print("🔧 训练优化建议:")

    if hardware.get('cuda_available'):
        total_memory = hardware.get('total_gpu_memory', 8)
        if total_memory < 8:
            print("- 考虑使用更小的max_length (64-96)")
            print("- 启用梯度累积以模拟更大的批次")
        else:
            print("- 使用混合精度训练减少内存使用")
            print("- 启用梯度检查点以节省内存")
            print("- 使用更大的批次大小充分利用GPU")

        print("- 监控GPU内存使用情况")
        print("- 定期清理GPU缓存")

    print("- 使用ModelScope加速模型下载")
    print("- 启用WandB监控训练过程")
    print("- 根据验证结果调整学习率")

if __name__ == "__main__":
    main()