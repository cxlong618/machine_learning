# 🖥️ Windows 部署指南

## 📋 概述

本指南详细介绍如何在Windows系统上部署产品分类模型，包括环境配置、模型部署和Web服务启动。

## 🎯 部署架构

```
Windows 10/11 部署架构:
┌─────────────────────────────────────┐
│           Web服务 (FastAPI)          │
│         端口: 8000                   │
├─────────────────────────────────────┤
│        产品分类推理器                │
│    - 模型文件: best_model.pt        │
│    - 分词器: tokenizer/             │
│    - 标签映射: label_mappings.json  │
├─────────────────────────────────────┤
│          系统要求                     │
│  - CPU: i5/i7 或 AMD Ryzen 5/7       │
│  - 内存: 16GB+ (推荐32GB)            │
│  - 存储: 50GB+ (推荐SSD)            │
│  - Python 3.8+                     │
└─────────────────────────────────────┘
```

## 🚀 快速部署

### 第一步：环境安装

1. **下载项目文件**
   ```
   下载整个产品分类项目文件夹到本地
   目录结构:
   产品分类项目/
   ├── models/              # 模型文件目录
   ├── src/                 # 源代码
   ├── run_inference.py     # 推理脚本
   ├── deploy_app.py        # Web服务
   └── install_windows.bat  # 安装脚本
   ```

2. **运行自动安装脚本**
   ```cmd
   # 双击运行或在命令行执行
   install_windows.bat
   ```

3. **手动安装（备选方案）**
   ```cmd
   # 创建虚拟环境
   python -m venv product_classifier_env

   # 激活环境
   product_classifier_env\Scripts\activate.bat

   # 安装依赖
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

### 第二步：准备模型文件

1. **从AutoDL下载模型文件**
   - 训练完成后，下载 `models/` 整个目录
   - 确保包含以下文件：
     ```
     models/
     ├── best_model.pt              # 最佳模型
     ├── label_mappings.json        # 标签映射
     └── tokenizer/                  # 分词器文件
         ├── vocab.json
         ├── special_tokens_map.json
         ├── added_tokens.json
         └── tokenizer_config.json
     ```

2. **文件放置**
   ```
   将下载的models目录放置在项目根目录:
   产品分类项目/models/
   ```

### 第三步：测试推理

1. **激活虚拟环境**
   ```cmd
   product_classifier_env\Scripts\activate.bat
   ```

2. **测试单个预测**
   ```cmd
   python run_inference.py --product "苹果iPhone手机"
   ```

3. **测试批量预测**
   ```cmd
   # 创建测试文件
   echo 苹果手机 > test_products.txt
   echo 华为笔记本 >> test_products.txt

   # 批量预测
   python run_inference.py --batch_input test_products.txt
   ```

4. **性能测试**
   ```cmd
   python run_inference.py --perf_test --perf_samples 100
   ```

### 第四步：启动Web服务

1. **启动服务**
   ```cmd
   python deploy_app.py
   ```

2. **访问服务**
   - API文档: http://localhost:8000/docs
   - 健康检查: http://localhost:8000/health
   - ReDoc文档: http://localhost:8000/redoc

3. **测试API接口**
   ```cmd
   # 使用curl测试
   curl -X POST "http://localhost:8000/classify" ^
        -H "Content-Type: application/json" ^
        -d "{\"product_name\": \"华为笔记本电脑\", \"return_prob\": true}"
   ```

## 📋 详细配置

### 系统要求检查

1. **硬件要求**
   ```cmd
   # 检查CPU
   wmic cpu get name

   # 检查内存
   wmic computersystem get totalphysicalmemory

   # 检查可用磁盘空间
   dir
   ```

2. **软件要求**
   ```cmd
   # 检查Python版本
   python --version

   # 检查pip版本
   pip --version

   # 检查CUDA（如果使用GPU）
   nvidia-smi
   ```

### 防火墙配置

1. **Windows防火墙设置**
   - 打开"Windows Defender 防火墙"
   - 点击"允许应用或功能通过Windows Defender防火墙"
   - 添加Python程序到允许列表
   - 允许端口8000的入站连接

2. **企业网络环境**
   ```
   如果在企业网络中，可能需要:
   - 联系IT部门开放端口8000
   - 配置代理设置
   - 使用HTTPS部署
   ```

### 性能优化

1. **内存优化**
   ```python
   # 在deploy_app.py中调整worker数量
   uvicorn.run(
       "deploy_app:app",
       host="0.0.0.0",
       port=8000,
       workers=1,  # Windows通常用1个worker
       reload=False,
       limit_concurrency=50
   )
   ```

2. **启动脚本优化**
   ```cmd
   # 创建高性能启动脚本 start_server.bat
   @echo off
   call product_classifier_env\Scripts\activate.bat
   set PYTHONPATH=%CD%
   python deploy_app.py --host 0.0.0.0 --port 8000 --workers 1
   ```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. Python环境问题
```
问题: 'python' 不是内部或外部命令
解决:
1. 重新安装Python，勾选"Add Python to PATH"
2. 或使用完整路径: C:\Python39\python.exe
3. 重启命令提示符
```

#### 2. 虚拟环境问题
```
问题: 无法激活虚拟环境
解决:
1. 使用完整路径激活
2. 检查Scripts目录是否存在
3. 重新创建虚拟环境
```

#### 3. 模型加载失败
```
问题: 模型文件不存在或加载失败
解决:
1. 检查models目录结构
2. 确认所有必需文件都存在
3. 检查文件路径中的中文和空格
```

#### 4. 端口被占用
```
问题: 端口8000已被使用
解决:
1. 查找占用进程: netstat -ano | findstr :8000
2. 结束进程: taskkill /PID <进程ID> /F
3. 或使用其他端口: python deploy_app.py --port 8001
```

#### 5. 内存不足
```
问题: 推理速度慢或内存不足
解决:
1. 减少批处理大小
2. 关闭不必要的程序
3. 升级内存到32GB或更多
4. 考虑使用GPU推理版本
```

#### 6. 中文字体显示问题
```
问题: API返回的中文显示乱码
解决:
1. 确保请求头包含: "Content-Type: application/json; charset=utf-8"
2. 检查终端编码设置
3. 使用支持UTF-8的客户端工具
```

### 日志调试

1. **启用详细日志**
   ```python
   # 在deploy_app.py中修改日志级别
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **保存日志到文件**
   ```cmd
   python deploy_app.py > server.log 2>&1
   ```

3. **监控服务状态**
   ```cmd
   # 检查进程
   tasklist | findstr python

   # 检查端口
   netstat -ano | findstr :8000
   ```

## 📊 性能基准

### 预期性能指标

基于CPU推理的性能基准：
```
硬件配置: i7-10700, 32GB RAM
├── 单次预测: 200-500ms
├── 批量预测(10个): 1-2秒
├── 吞吐量: 2-5 QPS
├── 内存占用: 2-4GB
└── CPU使用率: 30-50%
```

### 性能测试方法

1. **内置性能测试**
   ```cmd
   python run_inference.py --perf_test --perf_samples 1000
   ```

2. **API压力测试**
   ```cmd
   # 安装测试工具
   pip install locust

   # 创建测试脚本
   locust -f stress_test.py --host=http://localhost:8000
   ```

## 🔒 安全配置

### 生产环境安全建议

1. **网络安全**
   ```python
   # 在生产环境中使用HTTPS
   uvicorn.run(
       app,
       host="0.0.0.0",
       port=8000,
       ssl_keyfile="path/to/key.pem",
       ssl_certfile="path/to/cert.pem"
   )
   ```

2. **访问控制**
   ```python
   # 添加认证中间件
   from fastapi import Depends, HTTPBearer
   from fastapi.security import HTTPBearer

   security = HTTPBearer()

   @app.get("/secure-endpoint")
   async def secure_endpoint(token: str = Depends(security)):
       # 验证token
       return {"message": "认证成功"}
   ```

3. **请求限制**
   ```python
   # 安装依赖
   pip install slowapi

   # 添加限流中间件
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(_rate_limit_exceeded_handler, status_code=429)

   @app.post("/classify")
   @limiter.limit("10/minute")
   async def classify_rate_limited(request: ProductRequest):
       # 实现
       pass
   ```

## 🚀 高级部署选项

### Docker部署（可选）

1. **创建Dockerfile**
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app
   COPY . /app

   RUN pip install -r requirements.txt

   EXPOSE 8000
   CMD ["uvicorn", "deploy_app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **构建和运行**
   ```cmd
   docker build -t product-classifier .
   docker run -p 8000:8000 product-classifier
   ```

### 系统服务部署

1. **创建Windows服务**
   ```cmd
   # 使用NSSM (Non-Sucking Service Manager)
   nssm install "ProductClassifierAPI" python
   nssm set "ProductClassifierAPI" Arguments "deploy_app.py"
   nssm set "ProductClassifierAPI" DisplayName "产品分类API服务"
   nssm set "ProductClassifierAPI" StartType AUTO
   ```

## 📞 技术支持

### 联系方式
- 日志文件: `logs/training.log`
- 性能日志: `logs/performance.log`
- 错误报告: 检查控制台输出

### 常用命令
```cmd
# 检查服务状态
curl http://localhost:8000/health

# 查看API文档
start http://localhost:8000/docs

# 重启服务
# 停止服务后重新运行python deploy_app.py

# 清理虚拟环境
rmdir /s product_classifier_env
```

---

## 🎉 部署完成清单

- [ ] 环境安装完成 ✅
- [ ] 模型文件下载完成 ✅
- [ ] 推理测试通过 ✅
- [ ] Web服务启动成功 ✅
- [ ] API接口测试通过 ✅
- [ ] 防火墙配置完成 ✅
- [ ] 性能测试完成 ✅
- [ ] 监控日志正常 ✅

完成以上所有项目后，您的产品分类系统就成功部署在Windows上了！🎊