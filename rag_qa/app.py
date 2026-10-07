"""FastAPI 服务入口。

启动（在项目根目录 LocalRAG-QA/ 下）：
    uvicorn rag_qa.app:app --host 0.0.0.0 --port 8000

测试：
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d '{"question":"七天无理由退货运费谁出？"}'

工程化要点：
- 启动时一次性构建索引到内存（lifespan），请求阶段只做检索，避免重复建索引。
- Pydantic 模型校验入参，接口返回结构化 JSON，便于前端/第三方接入。
"""
import os
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import load_config
from .pipeline import RAGPipeline

_BASE = Path(__file__).resolve().parent          # rag_qa/ 目录
_ROOT = _BASE.parent                             # 项目根目录
_config = load_config()
_pipeline = RAGPipeline(_config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = _pipeline.build()
    print(f"[startup] 索引构建完成：{n} 个片段，耗时 {_pipeline._build_ms:.1f} ms")
    # 双击 start_web.bat 启动时自动打开浏览器
    if os.getenv("RAG_OPEN_BROWSER") == "1":
        webbrowser.open("http://127.0.0.1:8000")
    yield


app = FastAPI(title="LocalRAG-QA · 本地私有化 RAG 知识库问答系统", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索数量")


class AskResponse(BaseModel):
    answer: str
    hit: bool
    citations: list[dict]
    max_score: float
    coverage: float
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _pipeline.llm.model if _pipeline.llm.enabled else "抽取式演示(未配置大模型)",
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_ROOT / "static" / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return _pipeline.ask(req.question, top_k=req.top_k)