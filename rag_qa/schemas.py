"""接口层 Pydantic 模型：入参校验 + 出参结构。

FastAPI 的约定：请求体/响应体都用 Pydantic 模型定义，
框架会自动完成「JSON ↔ Python 对象」的转换与校验（类型不符直接 422）。
"""
from pydantic import BaseModel, Field


# ---------- 认证 ----------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, description="用户名")
    password: str = Field(..., min_length=6, max_length=64, description="密码")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


# ---------- 问答 ----------
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索数量")


class AnswerResponse(BaseModel):
    """问答接口的统一出参（登录版 /api/ask 与公开版 /ask 共用）。"""

    answer: str
    hit: bool
    citations: list[dict]
    max_score: float
    coverage: float
    latency_ms: float


class HistoryItem(BaseModel):
    """历史记录单条出参。"""

    id: int
    question: str
    answer: str
    sources: list[dict]
    hit: bool
    latency_ms: float
    created_at: str
