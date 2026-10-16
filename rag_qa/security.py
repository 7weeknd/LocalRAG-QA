"""安全模块：密码哈希（bcrypt）+ JWT 签发/校验 + 登录态依赖。

三个核心问题（面试讲清这里就赢了一半）：
1. 密码为什么不能明文存？
   数据库一旦泄露，明文密码直接暴露。bcrypt 是「慢哈希 + 自动加盐」：
   每份密码的盐随机生成并嵌在结果串里，相同密码两次哈希结果也不同，
   「慢」让暴力破解代价极高（GPU 每秒只能试几千次）。
2. 登录态怎么保持？
   HTTP 本身无状态，所以登录成功后签发一个 JWT（签过名的 JSON），
   前端存在 localStorage，之后每个请求放 Authorization 头带过来；
   服务端验签即可识别用户，不需要服务端存会话。
3. token 被人偷了怎么办？
   JWT 带过期时间（exp），超时自动失效；偷到 token 就像偷到一把会过期的钥匙。
"""
import datetime as dt

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import load_config
from .database import get_db
from .models import User

_config = load_config()

# auto_error=False：请求没带 Authorization 头时不在这里直接抛错，
# 由 get_current_user 统一返回 401 提示，错误信息更友好
_bearer = HTTPBearer(auto_error=False)


# ---------- 密码哈希 ----------
def hash_password(plain: str) -> str:
    """明文密码 → bcrypt 哈希（盐自动随机生成，结果里自带盐）。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码与库中哈希是否匹配。"""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:  # 库中哈希格式异常（如历史遗留脏数据），一律视为不匹配
        return False


# ---------- JWT ----------
def create_access_token(user: User) -> str:
    """为用户签发 token：payload 携带用户名与用户 id，附带过期时间。"""
    expire = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        minutes=_config.jwt_expire_minutes
    )
    payload = {"sub": user.username, "uid": user.id, "exp": expire}
    return jwt.encode(payload, _config.jwt_secret, algorithm="HS256")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI 依赖：解析 Authorization: Bearer <token>，返回对应 User。

    任何接口只要在函数参数里写上 `user: User = Depends(get_current_user)`，
    就自动获得「必须登录才能调用」的保护，同时拿到当前用户对象。
    """
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未登录，请先登录")
    try:
        payload = jwt.decode(
            credentials.credentials, _config.jwt_secret, algorithms=["HS256"]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录凭证无效")

    user = db.get(User, payload.get("uid"))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user