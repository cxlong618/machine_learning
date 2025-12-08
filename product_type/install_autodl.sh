#!/bin/bash
# AutoDL GPU环境安装脚本
# Python 3.12 + PyTorch 2.3.0 + CUDA 12.1

echo "🚀 AutoDL环境安装 - 产品分类模型"
echo "Python 3.12 + PyTorch 2.3.0 + CUDA 12.1"
echo "========================================="

# 检查CUDA版本
echo "🔍 检查系统环境..."
echo "CUDA版本:"
nvcc --version 2>/dev/null || echo "CUDA未检测到"
echo ""
echo "Python版本:"
python --version
echo ""

# 检查Python版本
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "⚠️  检测到Python $PYTHON_VERSION，建议使用Python 3.12"
    echo "AutoDL通常有多个Python版本，可以使用python3.12"
    echo ""
    # 检查是否有python3.12
    if command -v python3.12 &> /dev/null; then
        echo "✅ 发现python3.12，将使用python3.12"
        PYTHON_CMD="python3.12"
    else
        echo "❌ 未找到python3.12，将使用当前python"
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python"
    echo "✅ Python版本正确: 3.12"
fi

# 激活conda环境（如果存在）
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "🔄 激活conda环境..."
    source /root/miniconda3/bin/activate

    # 创建专用环境
    echo "📦 创建conda环境: product_classifier (Python 3.12)"
    conda create -n product_classifier python=3.12 -y
    conda activate product_classifier
    PYTHON_CMD="python"
else
    echo "📦 conda未安装，使用系统Python"
    PYTHON_CMD="python3"
fi

# 升级pip
echo "⬆️ 升级pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# 安装PyTorch 2.3.0 with CUDA 12.1
echo "🔥 安装PyTorch 2.3.0 (CUDA 12.1)..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch安装
echo "🧪 验证PyTorch安装..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 安装Transformers和NLP库
echo "🤗 安装Transformers和NLP库..."
pip install transformers==4.41.0
pip install tokenizers==0.19.0
pip install datasets==2.20.0
pip install jieba==0.42.1

# 安装数据处理库
echo "📊 安装数据处理库..."
pip install pandas==2.2.0
pip install numpy==1.26.0
pip install scikit-learn==1.4.0
pip install scipy==1.13.0

# 安装可视化库
echo "📈 安装可视化库..."
pip install matplotlib==3.8.0
pip install seaborn==0.13.0

# 安装日志和监控
echo "📝 安装日志和监控库..."
pip install tensorboard==2.17.0
pip install wandb==0.17.0
pip install tqdm==4.66.0

# 安装模型加速库
echo "⚡ 安装模型加速库..."
pip install accelerate==0.30.0

# 安装ModelScope模型库
echo "🏭 安装ModelScope模型库..."
pip install modelscope==1.15.1

# 安装其他依赖
echo "🔧 安装其他依赖..."
pip install pyyaml==6.0.1
pip install python-dotenv==1.0.0
pip install pathlib2==2.3.7
pip install colorama==0.4.6
pip install psutil==5.9.6

# 安装Web服务框架（用于本地测试）
echo "🌐 安装Web服务框架..."
pip install fastapi==0.111.0
pip install uvicorn==0.24.0

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p data models src config logs scripts docs

echo ""
echo "✅ 环境安装完成!"
echo ""
echo "📋 环境信息:"
$PYTHON_CMD --version
pip list | grep torch
pip list | grep transformers
echo ""

echo "🚀 现在可以开始训练:"
echo "   1. 上传数据文件到 data/ 目录"
echo "   2. 运行预处理: python scripts/data_preprocess.py --input data.xlsx"
echo "   3. 开始训练: python run_training.py --train_path data/train.csv --val_path data/val.csv"
echo ""

echo "💡 提示:"
echo "   - 训练过程日志保存在 logs/ 目录"
echo "   - 模型文件保存在 models/ 目录"
echo "   - 使用 nvidia-smi 监控GPU状态"
echo "   - 训练完成后下载 models/ 目录到本地"
echo ""

# 检查GPU状态
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU状态:"
    nvidia-smi
else
    echo "⚠️  nvidia-smi不可用，但CUDA库应该已安装"
fi

# 内存使用情况
echo "💾 内存使用:"
free -h

# 磁盘空间
echo "💿 磁盘空间:"
df -h /root

echo "🎉 安装脚本执行完成！"