"""ORM 数据模型：用户表 + 对话历史表。

表设计思路（求职面试高频问题，「为什么这么设计」都能答）：
- users：用户名唯一索引（登录查重/查找都走索引），密码只存 bcrypt 哈希绝不明文。
- chat_history：一条记录 = 一轮问答。sources 存「引用来源」的 JSON 字符串，
  保留检索证据，方便前端回显和历史复盘；user_id 外键把记录归属到具体用户。
- 外键 ondelete="CASCADE"：删除用户时自动清掉他的历史记录，不留孤儿数据。
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)  # 引用来源(JSON 字符串)
    hit: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class CopyHistory(Base):
    """AI 文案生成记录（AIGC 模块）。

    一条记录 = 一次文案生成：输入（产品/风格/语气/关键词）+ 输出（文案正文）。
    把输入也存下来，方便以后「复用同样的条件再生成一次」。
    """

    __tablename__ = "copy_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    product: Mapped[str] = mapped_column(String(200), nullable=False)
    style: Mapped[str] = mapped_column(String(50), nullable=False)
    tone: Mapped[str] = mapped_column(String(20), nullable=False)
    keywords: Mapped[str | None] = mapped_column(String(200), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )