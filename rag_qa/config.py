"""配置模块。

所有配置项都可通过环境变量覆盖，方便「本地跑通」与「接入真实大模型」两套场景切换：
- 不开环境变量：走内置 TF-IDF 检索 + 抽取式回答（零依赖、开箱即跑）
- 设置 LLM_* ：接入任意 OpenAI 兼容大模型（DeepSeek / 豆包 / 通义 / Ollama）
- 设置 MYSQL_* / JWT_* ：启用全栈版（注册登录 + 对话历史入库）

优先级：先加载项目根目录 .env 文件，再被真实环境变量覆盖。
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（rag_qa/ 的上一级），.env 就放在这里
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


@dataclass
class Config:
    # 知识库文档目录（.md / .txt）
    data_dir: str = "data/docs"

    # 切片参数：chunk_size 单块最大字符数，overlap 相邻块重叠字符数
    chunk_size: int = 300
    overlap: int = 50

    # 检索参数
    top_k: int = 5              # 每个检索器召回数量
    final_k: int = 3            # 融合后最终送入 Prompt 的片段数
    rrf_k: int = 60             # RRF 融合常数

    # 防幻觉阈值：相关性 = 最高余弦相似度 或 问题词覆盖率
    min_score: float = 0.05     # 最高余弦相似度下限
    min_coverage: float = 0.35  # 问题词覆盖率下限

    # 是否启用 jieba 分词（无 jieba 时自动降级）
    use_jieba: bool = True

    # 大模型配置（OpenAI 兼容 /chat/completions 接口）
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # MySQL 配置（全栈版：用户体系 + 对话历史入库）
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "rag_qa"

    # JWT 配置（登录令牌）
    jwt_secret: str = "dev-secret-change-me"  # 生产环境务必改成随机长串
    jwt_expire_minutes: int = 60 * 24 * 7     # 令牌有效期：7 天


def load_config() -> Config:
    """从环境变量读取配置，未设置时使用默认值。"""
    return Config(
        data_dir=os.getenv("RAG_DATA_DIR", "data/docs"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "300")),
        overlap=int(os.getenv("RAG_OVERLAP", "50")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        final_k=int(os.getenv("RAG_FINAL_K", "3")),
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.05")),
        min_coverage=float(os.getenv("RAG_MIN_COVERAGE", "0.35")),
        use_jieba=os.getenv("RAG_USE_JIEBA", "1") == "1",
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        db_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        db_port=int(os.getenv("MYSQL_PORT", "3306")),
        db_user=os.getenv("MYSQL_USER", "root"),
        db_password=os.getenv("MYSQL_PASSWORD", ""),
        db_name=os.getenv("MYSQL_DB", "rag_qa"),
        jwt_secret=os.getenv("JWT_SECRET", "dev-secret-change-me"),
        jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7))),
    )