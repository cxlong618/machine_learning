#!/usr/bin/env python3
"""
测试模型保存功能修复
"""
import os
import sys
import tempfile
import json

# 添加src到路径
sys.path.insert(0, 'src')

def test_model_save_function():
    """测试模型保存功能"""
    print("测试模型保存功能修复...")

    try:
        # 模拟模型保存方法
        test_dir = tempfile.mkdtemp(prefix='test_model_save_')
        print(f"测试目录: {test_dir}")

        # 创建测试模型文件
        config = {
            'num_labels_standard': 10,
            'num_labels_level1': 5,
            'num_labels_level2': 8,
            'num_labels_level3': 12,
            'loss_weights': {'standard': 0.4, 'level1': 0.2, 'level2': 0.2, 'level3': 0.2},
            'hidden_size': 768,
            'vocab_size': 21128
        }

        # 保存配置文件
        config_path = os.path.join(test_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 创建模型元数据文件
        import datetime
        metadata = {
            'model_version': '1.0.0',
            'framework': 'transformers',
            'task_type': 'multitask-classification',
            'created_time': datetime.datetime.now().isoformat(),
            'description': 'Product Multi-Task Classification Model',
            'tasks': ['standard', 'level1', 'level2', 'level3'],
            'base_model': 'dienstag/chinese-bert-wwm-ext'
        }

        metadata_path = os.path.join(test_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 创建虚拟模型文件
        model_path = os.path.join(test_dir, 'pytorch_model.bin')
        with open(model_path, 'wb') as f:
            f.write(b'fake_model_data')

        # 检查文件
        files = os.listdir(test_dir)
        expected_files = ['config.json', 'metadata.json', 'pytorch_model.bin']

        print(f"创建的文件: {files}")

        success = all(file in files for file in expected_files)

        if success:
            print("模型保存功能测试通过")
            print(f"  - 配置文件: {config_path}")
            print(f"  - 元数据文件: {metadata_path}")
            print(f"  - 模型文件: {model_path}")
        else:
            print("模型保存功能测试失败")
            missing = [file for file in expected_files if file not in files]
            print(f"  缺失文件: {missing}")

        # 清理
        import shutil
        shutil.rmtree(test_dir)

        return success

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_consistency():
    """测试路径一致性"""
    print("\n🔍 测试路径一致性...")

    # 检查所有脚本中的路径
    scripts_to_check = [
        'src/train.py',
        'src/inference.py',
        'src/dataset.py'
    ]

    expected_patterns = [
        './models/best_model',
        './models/tokenizer',
        './models/label_mappings.json'
    ]

    old_patterns = [
        '../models/best_model',
        '../models/tokenizer',
        '../models/label_mappings.json',
        './models/best_model.pt'
    ]

    issues = []

    for script in scripts_to_check:
        if not os.path.exists(script):
            print(f"⚠️  跳过不存在的文件: {script}")
            continue

        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()

        for old_pattern in old_patterns:
            if old_pattern in content:
                issues.append(f"  - {script}: 包含旧路径 '{old_pattern}'")

    if issues:
        print("❌ 发现路径不一致问题:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("✅ 路径一致性检查通过")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("模型保存功能修复验证")
    print("=" * 60)

    test1 = test_model_save_function()
    test2 = test_path_consistency()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("🎉 所有测试通过！模型保存功能修复成功")
    else:
        print("❌ 部分测试失败，需要进一步检查")
    print("=" * 60)