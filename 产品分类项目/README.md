# 🏭 产品分类项目

## 项目概述
基于BERT的多任务文本分类模型，通过产品名称同时预测：
- 标准名称（936类）
- 一级分类（24类）
- 二级分类（78类）
- 三级分类（138类）

## 数据信息
- **样本数量**: 456,732行
- **产品名称**: 81,109个唯一项
- **标准名称**: 936个类别
- **分类层次**: 一级24项 → 二级78项 → 三级138项

## 🚀 快速开始

### AutoDL GPU环境 (推荐)
```bash
# 1. 上传项目到AutoDL
git clone [项目地址]
cd 产品分类项目

# 2. 运行环境安装 (Python 3.12 + PyTorch 2.3.0 + CUDA 12.1)
bash install_autodl.sh

# 3. 数据预处理 (如果需要)
python scripts/data_preprocess.py --input data.xlsx

# 4. 开始训练
python run_training.py --train_path data/train.csv --val_path data/val.csv
```

### 本地环境
```bash
# Python 3.12环境
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 运行训练
python run_training.py --train_path data/train.csv --val_path data/val.csv
```

## 📋 项目结构
```
产品分类项目/
├── 📄 README.md                 # 项目说明
├── 📄 requirements.txt          # 依赖包 (Python 3.12 + PyTorch 2.3.0 + CUDA 12.1)
├── 📄 run_training.py          # 一键训练脚本
├── 📄 run_inference.py         # 推理脚本
├── 📄 deploy_app.py           # Windows部署应用
├── 📄 install_autodl.sh       # AutoDL环境安装
├── 📂 src/                    # 源代码
│   ├── 🐍 model.py            # 多任务BERT模型
│   ├── 🐍 dataset.py          # 数据处理
│   ├── 🐍 train.py             # 训练逻辑
│   ├── 🐍 inference.py         # 推理接口
│   └── 🐍 utils.py             # 工具函数
├── 📂 data/                   # 数据目录
├── 📂 models/                 # 模型目录
├── 📂 config/                 # 配置文件
│   └── 🐍 model_config.yaml    # 模型配置
├── 📂 logs/                   # 训练日志
├── 📂 scripts/               # 辅助脚本
│   └── 🐍 data_preprocess.py  # 数据预处理
└── 📂 docs/                  # 文档
    └── 📄 Windows部署指南.md  # 部署指南
```

## 🔧 技术架构

### 模型架构
- **基础模型**: chinese-bert-wwm-ext
- **多任务头**: 4个并行的分类器
- **损失函数**: 加权交叉熵损失
- **优化器**: AdamW + 学习率调度

### 训练配置
- **Python版本**: 3.12
- **PyTorch版本**: 2.3.0
- **CUDA版本**: 12.1
- **批次大小**: 32
- **学习率**: 2e-5
- **训练轮数**: 10 epochs
- **GPU**: RTX 4090/5090
- **内存**: 16-24GB VRAM

## 📊 性能指标

### 训练性能 (AutoDL RTX 4090)
- **训练时间**: 6-12小时
- **内存使用**: 16-24GB VRAM
- **最终准确率**: 85%+ (预期90%+)
- **模型大小**: 400-800MB

### 推理性能
- **CPU预测**: 200-500ms
- **GPU预测**: 50-100ms
- **吞吐量**: 2-5 QPS
- **内存占用**: 2-4GB

## 📈 预期结果

### 分类准确率目标
- **标准名称**: >85%
- **一级分类**: >95%
- **二级分类**: >90%
- **三级分类**: >88%
- **综合准确率**: >90%

### 部署要求
- **系统**: Windows 10/11
- **CPU**: i5/i7 或 AMD Ryzen 5/7
- **内存**: 16GB+ (推荐32GB)
- **存储**: 50GB+ (推荐SSD)

## 🎯 训练流程

### 第一步：环境准备
```bash
# AutoDL环境
bash install_autodl.sh

# 检查环境
python --version  # 应该是 3.12
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 第二步：数据准备
```bash
# 数据预处理
python scripts/data_preprocess.py --input your_data.xlsx

# 检查数据
head data/train.csv
wc -l data/train.csv
```

### 第三步：开始训练
```bash
# 基础训练
python run_training.py \
    --train_path data/train.csv \
    --val_path data/val.csv \
    --num_epochs 10 \
    --batch_size 32

# 自定义参数训练
python run_training.py \
    --train_path data/train.csv \
    --val_path data/val.csv \
    --max_length 128 \
    --batch_size 64 \
    --learning_rate 3e-5 \
    --num_epochs 15
```

### 第四步：模型下载
```bash
# 训练完成后，下载models目录
# 包含：
# - best_model.pt (最佳模型)
# - label_mappings.json (标签映射)
# - tokenizer/ (分词器文件)
```

## 🔧 高级配置

### 自定义参数
```python
# 在config/model_config.yaml中修改参数
training:
  epochs: 15
  learning_rate: 3e-5
  batch_size: 64

tasks:
  standard_name:
    weight: 0.5
  level1:
    weight: 0.2
  level2:
    weight: 0.2
  level3:
    weight: 0.1
```

### 多GPU训练
```bash
# 使用所有GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
    --train_path data/train.csv \
    --val_path data/val.csv \
    --batch_size 64
```

## 📋 故障排除

### 常见问题

1. **CUDA版本不匹配**
```bash
# 检查CUDA版本
nvcc --version

# 安装对应版本的PyTorch
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

2. **内存不足**
```bash
# 减少批次大小
python run_training.py --batch_size 16

# 使用梯度累积
python run_training.py --gradient_accumulation_steps 2
```

3. **依赖冲突**
```bash
# 创建新环境
conda create -n product_classifier python=3.12
conda activate product_classifier
pip install -r requirements.txt
```

### 调试技巧

1. **查看训练日志**
```bash
tail -f logs/training.log
```

2. **GPU监控**
```bash
watch -n 1 nvidia-smi
```

3. **内存监控**
```bash
watch -n 1 free -h
```

## 🚀 Windows部署

### 训练完成后部署
1. 下载 `models/` 目录
2. 本地创建Python环境
3. 运行推理测试
4. 启动Web服务

详细步骤请参考 `docs/Windows部署指南.md`

### Windows环境清单
```cmd
# 推荐配置
- Python: 3.12
- PyTorch: 2.3.0 (CPU版本)
- 内存: 32GB
- CPU: i7-10700
- 系统: Windows 10/11

# 安装命令
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install pandas==2.2.0 transformers==4.41.0
pip install fastapi==0.111.0 uvicorn==0.24.0
pip install jieba==0.42.1
```

## 📞 支持

### 文档
- 📖 Windows部署指南: `docs/Windows部署指南.md`
- 📊 模型架构: `src/model.py`
- 🔧 配置文件: `config/model_config.yaml`

### 示例代码
- 🧪 数据预处理: `scripts/data_preprocess.py`
- 🤖 推理测试: `run_inference.py`
- 🌐 Web服务: `deploy_app.py`

---

**训练完成预期时间**: 6-12小时
**最终准确率**: 90%+
**部署响应时间**: <1秒 (CPU), <100ms (GPU)