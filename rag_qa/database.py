"""数据库连接模块：SQLAlchemy 引擎 + 会话工厂。

设计思路（一次讲清三层）：
1. init_db()：启动时「自动建库建表」——连上 MySQL 服务器，库不存在就 CREATE，
   再用 ORM 模型 create_all 建表。用户装好 MySQL 后无需手工执行任何 SQL。
2. engine：全局连接池（自动维护 TCP 连接复用，不用每次请求重新握手）。
3. get_db()：FastAPI 依赖注入，一个请求一个会话（Session），请求结束自动关闭，
   避免连接泄漏，也是 FastAPI + SQLAlchemy 的标准写法。
"""
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import load_config

_config = load_config()


class Base(DeclarativeBase):
    """所有 ORM 模型的基类：继承它，SQLAlchemy 才能识别并自动建表。"""


def _build_url(db_name: str | None = None) -> URL:
    """构造 MySQL 连接串（URL 对象会自动处理密码里的特殊字符转义）。"""
    return URL.create(
        "mysql+pymysql",  # 方言+驱动：pymysql 是纯 Python 的 MySQL 驱动
        username=_config.db_user,
        password=_config.db_password,
        host=_config.db_host,
        port=_config.db_port,
        database=db_name,  # 为 None 时只连服务器本身（用于建库）
        query={"charset": "utf8mb4"},
    )


def init_db() -> None:
    """建库 + 建表（幂等：已存在则跳过，可重复调用）。"""
    # 第 1 步：连到服务器（不指定库），创建数据库（不存在才建）
    server_engine = create_engine(_build_url(), pool_pre_ping=True)
    with server_engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{_config.db_name}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
        conn.commit()
    server_engine.dispose()

    # 第 2 步：用正式引擎按 ORM 模型建表
    from . import models  # noqa: F401  # 导入 models 以把表定义注册到 Base.metadata

    Base.metadata.create_all(engine)


# 正式引擎：带数据库名的连接池（注意：模块导入时就创建，但未发起真实连接，
# 真正的 TCP 连接在第一次使用时才建立，所以 MySQL 没启动也不会在导入时报错）
engine = create_engine(_build_url(_config.db_name), pool_pre_ping=True)

# 会话工厂：每调用一次 SessionLocal() 得到一个新的数据库会话
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI 依赖：为每个请求提供独立会话，用完自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()