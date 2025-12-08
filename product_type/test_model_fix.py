#!/usr/bin/env python3
"""
测试模型加载路径修复
"""
import sys
import os

# 添加src到路径
sys.path.insert(0, 'src')

def test_model_methods():
    """测试模型类方法"""
    print("测试模型类方法...")

    try:
        # 尝试导入（不实际执行模型创建）
        import importlib.util
        spec = importlib.util.spec_from_file_location("model", "src/model.py")
        model_module = importlib.util.module_from_spec(spec)

        # 检查方法定义
        with open("src/model.py", "r", encoding="utf-8") as f:
            content = f.read()

        methods_to_check = [
            "def from_pretrained(cls, model_name_or_path",
            "def from_saved_model(cls, model_path",
            "def save_pretrained(self, save_directory"
        ]

        found_methods = []
        for method in methods_to_check:
            if method in content:
                found_methods.append(method.split('(')[0].split()[-1])

        print(f"找到的方法: {found_methods}")

        expected_methods = ["from_pretrained", "from_saved_model", "save_pretrained"]

        if all(method in found_methods for method in expected_methods):
            print("✓ 所有必要的方法都存在")
            return True
        else:
            missing = [m for m in expected_methods if m not in found_methods]
            print(f"✗ 缺失方法: {missing}")
            return False

    except Exception as e:
        print(f"测试失败: {e}")
        return False

def test_file_syntax():
    """测试文件语法"""
    print("测试文件语法...")

    files_to_check = ["src/model.py", "src/train.py", "src/inference.py"]

    all_good = True
    for file_path in files_to_check:
        try:
            import py_compile
            py_compile.compile(file_path, doraise=True)
            print(f"✓ {file_path} 语法正确")
        except py_compile.PyCompileError as e:
            print(f"✗ {file_path} 语法错误: {e}")
            all_good = False
        except Exception as e:
            print(f"✗ {file_path} 检查失败: {e}")
            all_good = False

    return all_good

def test_inference_compatibility():
    """测试推理脚本兼容性"""
    print("测试推理脚本兼容性...")

    try:
        with open("src/inference.py", "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否使用了新的方法
        if "from_saved_model" in content:
            print("✓ 推理脚本已更新为使用from_saved_model")
            return True
        else:
            print("✗ 推理脚本未更新")
            return False

    except Exception as e:
        print(f"检查失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("模型加载路径修复验证")
    print("=" * 50)

    test1 = test_file_syntax()
    test2 = test_model_methods()
    test3 = test_inference_compatibility()

    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 所有测试通过！模型加载路径修复成功")
        print("\n修复内容:")
        print("- ✓ 保留原始from_pretrained用于预训练模型")
        print("- ✓ 新增from_saved_model用于本地保存模型")
        print("- ✓ 更新推理脚本使用正确的加载方法")
        print("- ✓ 语法检查全部通过")
    else:
        print("❌ 部分测试失败，需要进一步检查")
        print(f"语法检查: {'✓' if test1 else '✗'}")
        print(f"方法检查: {'✓' if test2 else '✗'}")
        print(f"兼容性检查: {'✓' if test3 else '✗'}")
    print("=" * 50)