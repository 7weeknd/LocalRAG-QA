"""业务路由：注册 / 登录 / 问答（自动落库）/ 对话历史。

全栈版的接口，统一挂在 /api 前缀下：
- POST /api/auth/register   注册（用户名查重 → 密码哈希 → 入库）
- POST /api/auth/login      登录（验密 → 签发 JWT）
- POST /api/ask             问答（登录后：检索生成 → 自动写入对话历史）
- GET  /api/chat/history    查询当前用户的对话历史（最新的在前）

注意：登录版问答走 /api/ask；app.py 里原来的 /ask 保持公开，
不登录也能问答（演示模式），只是不记录历史——两者互不干扰。
"""
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .models import ChatHistory, User
from .pipeline import get_pipeline
from .schemas import (
    AnswerResponse,
    AskRequest,
    HistoryItem,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from .security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api")


@router.post("/auth/register", status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """注册：用户名查重，密码 bcrypt 哈希后入库。"""
    exists = db.scalar(select(User).where(User.username == req.username))
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已被占用")

    user = User(username=req.username, password_hash=hash_password(req.password))
    db.add(user)
    db.commit()
    return {"message": "注册成功", "username": user.username}


@router.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """登录：验证用户名密码，成功则签发 JWT 令牌返回。"""
    user = db.scalar(select(User).where(User.username == req.username))
    if user is None or not verify_password(req.password, user.password_hash):
        # 不提示「用户不存在」还是「密码错误」，避免被用来探测已注册账号
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")

    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "username": user.username,
    }


@router.post("/ask", response_model=AnswerResponse)
def ask(
    req: AskRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """登录版问答：检索生成后自动把这一轮对话写入 chat_history（服务端落库）。"""
    result = get_pipeline().ask(req.question, top_k=req.top_k)

    db.add(
        ChatHistory(
            user_id=user.id,  # 从 token 解析出的用户，前端无需、也无法伪造
            question=req.question,
            answer=result["answer"],
            sources=json.dumps(result["citations"], ensure_ascii=False),
            hit=result["hit"],
            latency_ms=result["latency_ms"],
        )
    )
    db.commit()
    return result


@router.get("/chat/history", response_model=list[HistoryItem])
def history(
    limit: int = Query(50, ge=1, le=200, description="最多返回条数"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """查询当前用户最近的对话历史（倒序：最新的一条在最前）。"""
    rows = db.scalars(
        select(ChatHistory)
        .where(ChatHistory.user_id == user.id)
        .order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc())
        .limit(limit)
    ).all()

    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "sources": json.loads(r.sources) if r.sources else [],
            "hit": r.hit,
            "latency_ms": r.latency_ms,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for r in rows
    ]
