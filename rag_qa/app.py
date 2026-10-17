"""FastAPI 服务入口。

启动（在项目根目录 LocalRAG-QA/ 下）：
    uvicorn rag_qa.app:app --host 0.0.0.0 --port 8010

网页入口：http://127.0.0.1:8010 （注册登录后在线问答，历史自动入库）

工程化要点：
- 启动时一次性构建索引到内存（lifespan），请求阶段只做检索，避免重复建索引。
- 启动时自动连接 MySQL 建库建表（不存在才建），免手工执行 SQL。
- 公开的 /ask 保持不登录可用（演示模式）；登录版 /api/ask 会自动写对话历史。
"""
import os
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from . import database
from .config import load_config
from .pipeline import get_pipeline
from .routes import router as api_router
from .schemas import AnswerResponse, AskRequest

_BASE = Path(__file__).resolve().parent          # rag_qa/ 目录
_ROOT = _BASE.parent                             # 项目根目录
_config = load_config()
_pipeline = get_pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = _pipeline.build()
    print(f"[startup] 索引构建完成：{n} 个片段，耗时 {_pipeline._build_ms:.1f} ms")

    # 自动建库建表；MySQL 没启动时报错但不退出，纯问答功能仍可用
    try:
        database.init_db()
        print(
            f"[startup] MySQL 连接成功：库 {_config.db_name} "
            "与表 users / chat_history 已就绪"
        )
    except Exception as exc:
        print(f"[warn] MySQL 初始化失败：{exc}")
        print("[warn] 注册登录 / 对话历史暂不可用（请先启动 MySQL），纯问答不受影响")

    # 双击 start_web.bat 启动时自动打开浏览器
    if os.getenv("RAG_OPEN_BROWSER") == "1":
        webbrowser.open("http://127.0.0.1:8010")
    yield


app = FastAPI(
    title="LocalRAG-QA · 本地私有化 RAG 知识库问答系统",
    version="0.3.0",
    lifespan=lifespan,
)

# 注册登录 / 问答落库 / 历史查询（见 routes.py）
app.include_router(api_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _pipeline.llm.model if _pipeline.llm.enabled else "抽取式演示(未配置大模型)",
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_ROOT / "static" / "index.html")


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest):
    """公开问答接口（无需登录，不记录历史）：保留给命令行 / 三方快速接入。"""
    return _pipeline.ask(req.question, top_k=req.top_k)