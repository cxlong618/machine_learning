#!/usr/bin/env python3
"""
产品分类Web部署应用
基于FastAPI的RESTful API服务
"""
import os
import sys
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import time
import json
from datetime import datetime

from inference import get_inference_instance

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="产品分类API",
    description="基于BERT的多任务产品分类系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
inference = None
startup_time = datetime.now()


# 请求模型
class ProductRequest(BaseModel):
    product_name: str = Field(..., description="产品名称", min_length=1, max_length=200)
    return_prob: bool = Field(True, description="是否返回置信度")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Top-K预测数量")


class ProductResponse(BaseModel):
    success: bool = Field(..., description="请求是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="预测结果")
    error: Optional[str] = Field(None, description="错误信息")
    response_time: str = Field(..., description="响应时间(ms)")
    timestamp: str = Field(..., description="时间戳")


class BatchRequest(BaseModel):
    products: List[str] = Field(..., description="产品名称列表", min_items=1, max_items=100)
    return_prob: bool = Field(True, description="是否返回置信度")


class BatchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_products: int
    total_time: str
    avg_time: str
    errors: List[str]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: str
    system_info: Dict[str, Any]


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化推理器"""
    global inference

    try:
        logger.info("🚀 启动产品分类服务...")
        inference = get_inference_instance()
        logger.info("✅ 推理器加载成功")
    except Exception as e:
        logger.error(f"❌ 推理器加载失败: {e}")
        inference = None


@app.get("/", tags=["基础"])
async def root():
    """根路径"""
    return {
        "message": "产品分类API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    uptime = datetime.now() - startup_time
    hours, remainder = divmod(uptime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    uptime_str = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"

    # 获取系统信息
    try:
        import psutil
        import torch
        import platform

        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1)
        }

        if torch.cuda.is_available():
            system_info["gpu_name"] = torch.cuda.get_device_name()
            system_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)

    except ImportError:
        system_info = {"message": "系统信息模块未完全可用"}

    return HealthResponse(
        status="healthy" if inference else "degraded",
        model_loaded=inference is not None,
        uptime=uptime_str,
        system_info=system_info
    )


@app.post("/classify", response_model=ProductResponse, tags=["分类"])
async def classify_product(request: ProductRequest):
    """单个产品分类"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    start_time = time.time()

    try:
        # 调用推理
        result = inference.predict(request.product_name, request.return_prob)

        if 'error' in result:
            response_time = f"{(time.time() - start_time) * 1000:.2f}"
            return ProductResponse(
                success=False,
                error=result['error'],
                response_time=response_time,
                timestamp=datetime.now().isoformat()
            )

        # Top-K预测
        if request.top_k:
            try:
                top_k_result = inference.get_top_k_predictions(request.product_name, request.top_k)
                result['top_k_predictions'] = top_k_result
            except Exception as e:
                logger.warning(f"Top-K预测失败: {e}")

        response_time = f"{(time.time() - start_time) * 1000:.2f}"

        return ProductResponse(
            success=True,
            data=result,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        response_time = f"{(time.time() - start_time) * 1000:.2f}"
        logger.error(f"分类失败: {e}")
        return ProductResponse(
            success=False,
            error=f"分类失败: {str(e)}",
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )


@app.post("/classify/batch", response_model=BatchResponse, tags=["分类"])
async def classify_batch(request: BatchRequest):
    """批量产品分类"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    start_time = time.time()

    try:
        results = []
        errors = []

        # 批量预测
        batch_results = inference.predict_batch(request.products, request.return_prob)

        for i, (product_name, result) in enumerate(zip(request.products, batch_results)):
            if 'error' in result:
                errors.append(f"产品 {i+1} ({product_name[:30]}...): {result['error']}")
            else:
                results.append(result)

        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(request.products)

        return BatchResponse(
            success=True,
            results=results,
            total_products=len(request.products),
            total_time=f"{total_time:.2f}ms",
            avg_time=f"{avg_time:.2f}ms",
            errors=errors,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"批量分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量分类失败: {str(e)}")


@app.post("/classify/top_k", tags=["分类"])
async def classify_top_k(product_name: str = Field(..., description="产品名称"),
                       k: int = Field(5, description="Top-K数量", ge=1, le=10)):
    """Top-K分类预测"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    try:
        result = inference.get_top_k_predictions(product_name, k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top-K预测失败: {str(e)}")


@app.get("/stats", tags=["系统"])
async def get_stats():
    """获取系统统计信息"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    try:
        # 运行性能测试
        perf_stats = inference.evaluate_performance(num_samples=50)

        return {
            "model_info": {
                "model_loaded": True,
                "model_path": inference.model_path
            },
            "performance": perf_stats,
            "service": {
                "uptime": str(datetime.now() - startup_time),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.get("/categories", tags=["信息"])
async def get_categories():
    """获取所有类别信息"""
    try:
        mapping_path = "./models/label_mappings.json"
        if not os.path.exists(mapping_path):
            raise HTTPException(status_code=404, detail="标签映射文件不存在")

        with open(mapping_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        return {
            "standard_name_categories": mappings['standard_mapping'],
            "level1_categories": mappings['level1_mapping'],
            "level2_categories": mappings['level2_mapping'],
            "level3_categories": mappings['level3_mapping'],
            "total_standard_names": len(mappings['standard_mapping']),
            "total_level1": len(mappings['level1_mapping']),
            "total_level2": len(mappings['level2_mapping']),
            "total_level3": len(mappings['level3_mapping'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取类别信息失败: {str(e)}")


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}")
    return {
        "error": True,
        "message": "内部服务器错误",
        "timestamp": datetime.now().isoformat()
    }


def main():
    """启动服务器"""
    print("🚀 启动产品分类API服务")
    print("="*50)
    print("📋 服务信息:")
    print("  - API文档: http://localhost:8000/docs")
    print("  - ReDoc文档: http://localhost:8000/redoc")
    print("  - 健康检查: http://localhost:8000/health")
    print("="*50)

    uvicorn.run(
        "deploy_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    main()