#!/usr/bin/env python3
"""
快速启动训练 - 自动处理CUDA问题
"""
import os
import sys

def main():
    print("🚀 快速训练启动器")
    print("=" * 50)

    # 设置强制CPU训练（避免CUDA问题）
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # 检查数据文件
    train_file = 'data/train.csv'
    val_file = 'data/val.csv'

    if not os.path.exists(train_file):
        print(f"❌ 训练文件不存在: {train_file}")
        return False

    if not os.path.exists(val_file):
        print(f"❌ 验证文件不存在: {val_file}")
        return False

    print("✅ 数据文件检查通过")
    print("🖥️  使用强制CPU训练模式（稳定但较慢）")
    print("📊 优化参数: 小批次 + 少epoch + 快速验证")

    # 构建训练命令
    cmd_parts = [
        'python', 'src/train.py',
        '--train_path', train_file,
        '--val_path', val_file,
        '--batch_size', '4',        # CPU使用小批次
        '--num_epochs', '2',        # 快速训练
        '--learning_rate', '5e-5',   # 稍高学习率
        '--max_length', '64',         # 减少序列长度
        '--warmup_steps', '50',        # 快速预热
    ]

    print(f"🔥 启动命令: {' '.join(cmd_parts)}")
    print("⚡ 这应该立即开始训练，避免CUDA问题！")
    print("=" * 50)

    # 启动训练
    try:
        import subprocess
        result = subprocess.run(cmd_parts, cwd='.', capture_output=True, text=True)

        print("📊 训练输出:")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("⚠️ 警告:")
            print(result.stderr)

        print(f"🏁 训练完成，退出码: {result.returncode}")

        if result.returncode == 0:
            print("✅ 训练成功完成!")
            print("📁 检查 ./models/ 目录获取训练结果")
            return True
        else:
            print("❌ 训练失败")
            return False

    except Exception as e:
        print(f"❌ 启动训练失败: {e}")
        return False

if __name__ == "__main__":
    main()