#!/usr/bin/env python3
"""
产品分类Web部署应用
基于FastAPI的RESTful API服务
"""
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# 添加src目录到Python路径
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.inference import get_inference_instance
import uvicorn
import time
import json
from datetime import datetime
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
inference = None
startup_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    global inference
    try:
        logger.info("🚀 启动产品分类服务...")
        inference = get_inference_instance()
        logger.info("✅ 推理器加载成功")
    except Exception as e:
        logger.error(f"❌ 推理器加载失败: {e}")
        inference = None

    yield

    # 关闭时执行（如果需要）
    logger.info("👋 关闭产品分类服务")


# 创建FastAPI应用
app = FastAPI(
    title="产品分类API",
    description="基于BERT的多任务产品分类系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加静态文件服务
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    app.logger.warning(f"静态文件目录不存在: {static_dir}")


# 请求模型
class ProductRequest(BaseModel):
    product_name: str = Field(..., description="产品名称", min_length=1, max_length=200)
    return_prob: bool = Field(True, description="是否返回置信度")


class ProductResponse(BaseModel):
    success: bool = Field(..., description="请求是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="预测结果")
    error: Optional[str] = Field(None, description="错误信息")
    response_time: str = Field(..., description="响应时间(ms)")
    timestamp: str = Field(..., description="时间戳")


class BatchRequest(BaseModel):
    products: List[str] = Field(..., description="产品名称列表", min_length=1, max_length=100)
    return_prob: bool = Field(True, description="是否返回置信度")


class TopKRequest(BaseModel):
    product_name: str = Field(..., description="产品名称", min_length=1, max_length=200)
    k: int = Field(5, description="Top-K数量", ge=1, le=10)


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


@app.get("/", tags=["基础"])
async def root():
    """根路径 - 重定向到前端页面"""
    if os.path.exists("static/index.html"):
        return HTMLResponse(open("static/index.html", encoding='utf-8').read())
    else:
        return {
            "message": "产品分类API服务",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    uptime = str(datetime.now() - startup_time)

    return {
        "status": "healthy",
        "model_loaded": inference is not None,
        "uptime": uptime,
        "system_info": {
            "version": "1.0.0",
            "model_loaded": inference is not None
        }
    }


@app.post("/classify", response_model=ProductResponse, tags=["分类"])
async def classify_product(request: ProductRequest):
    """单个产品分类"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    start_time = time.time()

    try:
        logger.info(f"收到分类请求: {request.product_name}")

        # 调用推理器
        result = inference.predict(
            request.product_name,
            return_prob=request.return_prob
        )

        response_time = (time.time() - start_time) * 1000

        logger.info(f"分类完成，结果: {result.get('standard_name', 'N/A')}")

        return ProductResponse(
            success=True,
            data=result,
            response_time=f"{response_time:.2f}",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"分类失败: {e}")
        return ProductResponse(
            success=False,
            error=str(e),
            response_time=f"{response_time:.2f}",
            timestamp=datetime.now().isoformat()
        )


@app.post("/classify/batch", response_model=BatchResponse, tags=["分类"])
async def classify_batch(request: BatchRequest):
    """批量产品分类"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    start_time = time.time()
    results = []
    errors = []

    for product_name in request.products:
        try:
            result = inference.predict(
                product_name,
                return_prob=request.return_prob
            )
            results.append({
                "product_name": product_name,
                "result": result,
                "success": True
            })
        except Exception as e:
            error_msg = f"产品 '{product_name}' 分类失败: {str(e)}"
            errors.append(error_msg)
            results.append({
                "product_name": product_name,
                "result": None,
                "success": False,
                "error": str(e)
            })

    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / len(request.products) if request.products else 0

    return BatchResponse(
        success=len(errors) == 0,
        results=results,
        total_products=len(request.products),
        total_time=f"{total_time:.2f}",
        avg_time=f"{avg_time:.2f}",
        errors=errors,
        timestamp=datetime.now().isoformat()
    )


@app.post("/classify/top_k", tags=["分类"])
async def classify_top_k(request: TopKRequest):
    """Top-K分类预测"""
    if not inference:
        raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

    try:
        result = inference.get_top_k_predictions(request.product_name, request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top-K预测失败: {str(e)}")


@app.get("/stats", tags=["系统"])
async def get_stats():
    """获取系统统计信息"""
    uptime = str(datetime.now() - startup_time)

    stats = {
        "uptime": uptime,
        "model_loaded": inference is not None,
        "timestamp": datetime.now().isoformat()
    }

    if inference:
        # 添加推理器统计信息（如果可用）
        stats["inference_stats"] = {
            "model_loaded": True
        }

    return stats


if __name__ == "__main__":
    import socket

    # 获取本机IP地址
    try:
        # 创建一个socket连接到公共DNS来获取本机IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"

    print("=" * 60)
    print("产品分类API服务启动中...")
    print(f"本地访问: http://localhost:8000")
    print(f"局域网访问: http://{local_ip}:8000")
    print(f"API文档: http://{local_ip}:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "deploy_app:app",
        host="0.0.0.0",  # 绑定到所有网络接口
        port=8000,
        reload=True,
        log_level="info"
    )