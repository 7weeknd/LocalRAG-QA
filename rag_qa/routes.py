"""业务路由：注册 / 登录 / 问答（自动落库）/ 对话历史 / AI 文案生成。

全栈版的接口，统一挂在 /api 前缀下：
- POST /api/auth/register   注册（用户名查重 → 密码哈希 → 入库）
- POST /api/auth/login      登录（验密 → 签发 JWT）
- POST /api/ask             问答（登录后：检索生成 → 自动写入对话历史）
- GET  /api/chat/history    查询当前用户的对话历史（最新的在前）
- POST /api/copy            AI 文案生成（调用大模型 → 写入文案记录）
- GET  /api/copy/history    查询当前用户的文案生成记录

注意：登录版问答走 /api/ask；app.py 里原来的 /ask 保持公开，
不登录也能问答（演示模式），只是不记录历史——两者互不干扰。
"""
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .copywriter import generate_copy
from .database import get_db
from .models import ChatHistory, CopyHistory, User
from .pipeline import get_pipeline
from .schemas import (
    AnswerResponse,
    AskRequest,
    CopyHistoryItem,
    CopyRequest,
    CopyResponse,
    CopyUpdateRequest,
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


# ---------- AI 文案生成（AIGC） ----------
@router.post("/copy", response_model=CopyResponse)
def copy(
    req: CopyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """AI 文案生成：调用大模型生成营销文案，并写入 copy_history。"""
    try:
        result = generate_copy(req.product, req.style, req.tone, req.keywords)
    except RuntimeError as exc:  # 未配置大模型
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except Exception as exc:  # 大模型调用失败（网络/超时/返回异常）
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"大模型调用失败：{exc}")

    row = CopyHistory(
        user_id=user.id,  # 归属用户由 token 解析，前端无法伪造
        product=req.product,
        style=req.style,
        tone=req.tone,
        keywords=req.keywords or None,
        content=result["content"],
        latency_ms=result["latency_ms"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)  # 取回自增 id 与数据库生成的 created_at

    return {
        "id": row.id,
        "product": row.product,
        "style": row.style,
        "tone": row.tone,
        "content": row.content,
        "latency_ms": row.latency_ms,
        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/copy/history", response_model=list[CopyHistoryItem])
def copy_history(
    limit: int = Query(20, ge=1, le=100, description="最多返回条数"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """查询当前用户最近的文案生成记录（倒序：最新的一条在最前）。"""
    rows = db.scalars(
        select(CopyHistory)
        .where(CopyHistory.user_id == user.id)
        .order_by(CopyHistory.created_at.desc(), CopyHistory.id.desc())
        .limit(limit)
    ).all()

    return [
        {
            "id": r.id,
            "product": r.product,
            "style": r.style,
            "content": r.content,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for r in rows
    ]


@router.delete("/chat/history/{id}")
def delete_history(
    id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除当前用户的一条对话记录（只能删自己的）。"""
    row = db.get(ChatHistory, id)  # 按主键取一条；id 不存在时返回 None
    if row is None or row.user_id != user.id:
        # 统一返回 404：不暴露「记录是否存在」，防止被用来探测别人的数据
        raise HTTPException(status.HTTP_404_NOT_FOUND, "记录不存在")

    db.delete(row)
    db.commit()  # 不 commit 只在内存里标记删除，不会真正落库
    return {"message": "已删除", "id": id}


@router.patch("/copy/history/{id}", response_model=CopyHistoryItem)
def update_copy(
    id: int,
    req: CopyUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """修改一条文案记录（只能改自己的）：把 AI 生成的文案改成自己想要的版本。"""
    row = db.get(CopyHistory, id)  # 按主键取一条
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "记录不存在")

    row.content = req.content  # 直接改对象属性 —— ORM 的 UPDATE 就是这么简单
    db.commit()  # 提交后才会生成 UPDATE 语句真正落库
    db.refresh(row)  # 重新读回，拿到数据库最新值

    return {
        "id": row.id,
        "product": row.product,
        "style": row.style,
        "content": row.content,
        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }