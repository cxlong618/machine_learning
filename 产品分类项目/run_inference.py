#!/usr/bin/env python3
"""
产品分类推理脚本
用于测试和运行产品分类模型推理
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from inference import ProductInference, get_inference_instance

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def single_prediction(inference, product_name: str, return_prob: bool = False):
    """单个产品预测"""
    print(f"\n🔍 产品分类预测")
    print("="*50)
    print(f"产品名称: {product_name}")
    print("-"*50)

    result = inference.predict(product_name, return_prob=return_prob)

    if 'error' in result:
        print(f"❌ 预测失败: {result['error']}")
        return

    print(f"✅ 预测结果:")
    print(f"  📦 标准名称: {result['standard_name']}")
    print(f"  📂 一级分类: {result['level1_category']}")
    print(f"  📁 二级分类: {result['level2_category']}")
    print(f"  📄 三级分类: {result['level3_category']}")

    if return_prob:
        print(f"\n🎯 置信度:")
        print(f"  标准名称: {result.get('confidence_standard', 0):.3f}")
        print(f"  一级分类: {result.get('confidence_level1', 0):.3f}")
        print(f"  二级分类: {result.get('confidence_level2', 0):.3f}")
        print(f"  三级分类: {result.get('confidence_level3', 0):.3f}")
        print(f"  综合置信度: {result.get('overall_confidence', 0):.3f}")

    print(f"⏱️  响应时间: {result.get('response_time', 'N/A')}")
    print(f"🔤 处理后文本: {result.get('processed_text', 'N/A')}")


def top_k_prediction(inference, product_name: str, k: int = 5):
    """Top-K预测"""
    print(f"\n🎯 Top-{k} 预测结果")
    print("="*50)
    print(f"产品名称: {product_name}")
    print("-"*50)

    result = inference.get_top_k_predictions(product_name, k=k)

    if 'error' in result:
        print(f"❌ 预测失败: {result['error']}")
        return

    for task in ['standard', 'level1', 'level2', 'level3']:
        if task in result:
            print(f"\n📊 {task.upper()} 分类 Top-{k}:")
            for i, item in enumerate(result[task], 1):
                prob = item['probability']
                bar_length = int(prob * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {i:2d}. {item['label']:<30} |{bar}| {prob:.4f}")


def batch_prediction(inference, input_file: str, output_file: str = None):
    """批量预测"""
    print(f"\n📦 批量预测")
    print("="*50)
    print(f"输入文件: {input_file}")

    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                # JSON格式
                data = json.load(f)
                if isinstance(data, list):
                    product_names = data
                elif isinstance(data, dict) and 'products' in data:
                    product_names = data['products']
                else:
                    raise ValueError("不支持的JSON格式")
            else:
                # 文本格式 (每行一个产品名称)
                product_names = [line.strip() for line in f if line.strip()]

        print(f"产品数量: {len(product_names)}")

        # 批量预测
        print("开始预测...")
        results = []
        for i, product_name in enumerate(product_names, 1):
            print(f"进度: {i}/{len(product_names)} - {product_name[:30]}...")
            result = inference.predict(product_name, return_prob=True)
            results.append(result)

        # 保存结果
        if output_file:
            output_data = {
                'input_file': input_file,
                'total_products': len(product_names),
                'predictions': results
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"✅ 结果已保存到: {output_file}")

        # 显示统计信息
        print("\n📊 预测统计:")
        if 'overall_confidence' in results[0]:
            confidences = [r['overall_confidence'] for r in results]
            print(f"  平均置信度: {sum(confidences)/len(confidences):.3f}")
            print(f"  最高置信度: {max(confidences):.3f}")
            print(f"  最低置信度: {min(confidences):.3f}")

    except Exception as e:
        print(f"❌ 批量预测失败: {e}")


def performance_test(inference, num_samples: int = 100):
    """性能测试"""
    print(f"\n⚡ 性能测试")
    print("="*50)
    print(f"测试样本数: {num_samples}")

    performance = inference.evaluate_performance(num_samples)

    print("\n📊 性能统计:")
    print(f"  📈 平均响应时间: {performance['avg_time_ms']:.2f} ms")
    print(f"  ⚡ 最快响应时间: {performance['min_time_ms']:.2f} ms")
    print(f"  🐌 最慢响应时间: {performance['max_time_ms']:.2f} ms")
    print(f"  📊 标准差: {performance['std_time_ms']:.2f} ms")
    print(f"  📈 中位数: {performance['median_time_ms']:.2f} ms")
    print(f"  📉 P95: {performance['p95_time_ms']:.2f} ms")
    print(f"  🚀 吞吐量: {performance['throughput_qps']:.2f} QPS")

    # 性能评估
    avg_time = performance['avg_time_ms']
    if avg_time < 100:
        print(f"  ✅ 性能优秀 ({avg_time:.1f}ms < 100ms)")
    elif avg_time < 500:
        print(f"  🟡 性能良好 ({avg_time:.1f}ms < 500ms)")
    elif avg_time < 1000:
        print(f"  🟠 性能一般 ({avg_time:.1f}ms < 1000ms)")
    else:
        print(f"  ❌ 性能较差 ({avg_time:.1f}ms > 1000ms)")


def interactive_mode(inference):
    """交互式模式"""
    print("\n🎮 交互式模式")
    print("="*50)
    print("输入产品名称进行预测，输入 'quit' 退出")
    print("输入 'top-k <k>' 进行Top-K预测")
    print("输入 'perf' 进行性能测试")
    print("-"*50)

    while True:
        try:
            user_input = input("\n产品名称 > ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break

            if user_input.lower() == 'perf':
                performance_test(inference)
                continue

            if user_input.lower().startswith('top-k'):
                parts = user_input.split()
                k = int(parts[1]) if len(parts) > 1 else 3
                product_name = input("请输入产品名称: ").strip()
                if product_name:
                    top_k_prediction(inference, product_name, k)
                continue

            if user_input:
                single_prediction(inference, user_input, return_prob=True)

        except KeyboardInterrupt:
            print("\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 处理失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='产品分类模型推理脚本')

    parser.add_argument('--model_path', type=str, default='./models/best_model.pt',
                       help='模型文件路径')
    parser.add_argument('--product', type=str, help='单个产品名称')
    parser.add_argument('--top_k', type=int, default=5, help='Top-K数量')
    parser.add_argument('--batch_input', type=str, help='批量预测输入文件')
    parser.add_argument('--batch_output', type=str, help='批量预测输出文件')
    parser.add_argument('--perf_test', action='store_true', help='性能测试')
    parser.add_argument('--perf_samples', type=int, default=100, help='性能测试样本数')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')

    args = parser.parse_args()

    print("🤖 产品分类模型推理工具")
    print("="*50)

    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        print("请确保:")
        print("  1. 模型训练已完成")
        print("  2. 模型文件已保存到正确位置")
        return 1

    # 创建推理器
    try:
        print("🔄 加载模型...")
        inference = ProductInference(args.model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1

    # 根据参数执行不同操作
    if args.product:
        # 单个预测
        single_prediction(inference, args.product, return_prob=True)

    elif args.top_k and args.product:
        # Top-K预测
        top_k_prediction(inference, args.product, args.top_k)

    elif args.batch_input:
        # 批量预测
        output_file = args.batch_output or f"{args.batch_input}_results.json"
        batch_prediction(inference, args.batch_input, output_file)

    elif args.perf_test:
        # 性能测试
        performance_test(inference, args.perf_samples)

    elif args.interactive:
        # 交互式模式
        interactive_mode(inference)

    else:
        # 默认：示例预测
        print("📝 运行示例预测...")
        examples = [
            "苹果iPhone 14 Pro手机",
            "华为MateBook X Pro笔记本电脑",
            "小米65寸智能电视"
        ]

        for example in examples:
            single_prediction(inference, example, return_prob=True)

        print(f"\n💡 使用说明:")
        print(f"  --product '产品名称'        : 单个产品预测")
        print(f"  --top_k 3 --product '名称'  : Top-K预测")
        print(f"  --batch_input file.txt     : 批量预测")
        print(f"  --perf_test                : 性能测试")
        print(f"  --interactive              : 交互式模式")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)