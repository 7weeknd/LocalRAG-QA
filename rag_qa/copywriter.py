"""AI 文案生成模块（AIGC）。

和 RAG 问答的区别：
- 问答要「忠于知识库、禁止编造」→ 提示词强调约束，temperature 用低温 0.2；
- 文案要「有创意、有感染力」→ 提示词强调风格与格式，temperature 用高温 0.8。

设计要点：
- STYLES：每种平台/体裁一套「格式要求」，用 dict 存；新增风格只需加一项，不改逻辑。
- build_prompts()：纯函数，把参数拼成 system + user 两条提示词，方便单独测试。
- generate_copy()：调用大模型；未配置 Key 时抛 RuntimeError，由路由层转成友好提示。
"""
import time

from .config import Config, load_config
from .llm import LLMClient

# 平台/体裁 → 展示名 + 格式要求（格式要求会拼进 user 提示词）
STYLES: dict[str, dict] = {
    "xiaohongshu": {
        "label": "小红书笔记",
        "format": (
            "先写一句抓眼球的标题（带 emoji），换行后正文分 3-5 行写卖点，"
            "每行开头用 emoji；结尾带 3-5 个话题标签（# 开头）。总字数 120-200。"
        ),
    },
    "moments": {
        "label": "朋友圈 / 社群",
        "format": (
            "口语化，像朋友真心推荐：1-2 句勾兴趣 + 3 个卖点短句，"
            "结尾一句行动号召。不要话题标签。总字数 60-120。"
        ),
    },
    "ecommerce": {
        "label": "电商商品详情",
        "format": (
            "先写一句话核心卖点，再列 5 条参数或优势（每条一行，用「-」开头），"
            "最后一句售后承诺。风格专业、信息密度高。总字数 100-180。"
        ),
    },
    "general": {
        "label": "通用营销文案",
        "format": "结构清晰：标题 + 正文 + 行动号召，总字数 100-150。",
    },
}

# 可选语气
TONES = ["活泼", "专业", "亲切", "高冷"]

_SYSTEM_PROMPT = (
    "你是一名资深中文营销文案撰稿人，擅长把产品卖点写得有感染力、贴合平台调性。"
    "只输出文案本身，不要输出任何解释、前言或「以下是」之类的引导语。"
)


def build_prompts(product: str, style: str, tone: str, keywords: str) -> tuple[str, str]:
    """把输入参数拼成 (system, user) 两条提示词。纯函数，无副作用。"""
    style_cfg = STYLES.get(style) or STYLES["general"]
    user = (
        f"产品 / 主题：{product}\n"
        f"平台风格：{style_cfg['label']}\n"
        f"语气：{tone}\n"
        f"必须包含的关键词：{keywords or '（无）'}\n\n"
        f"格式要求：{style_cfg['format']}"
    )
    return _SYSTEM_PROMPT, user


# 进程级单例：和 pipeline 的 LLM 客户端各自独立，互不影响
_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        cfg: Config = load_config()
        _llm = LLMClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    return _llm


def generate_copy(
    product: str, style: str = "xiaohongshu", tone: str = "活泼", keywords: str = ""
) -> dict:
    """生成一段营销文案，返回 {content, latency_ms}。

    未配置大模型 → 抛 RuntimeError（路由层转 503）；
    调用失败 → 由路由层捕获转 502。
    """
    llm = get_llm()
    if not llm.enabled:
        raise RuntimeError("未配置大模型，无法生成文案（请在 start_web.bat 或 .env 中配置 LLM_*）")

    system, user = build_prompts(product, style, tone, keywords)
    t0 = time.perf_counter()
    content = llm.generate(system, user, temperature=0.8)  # 高温求创意
    latency = (time.perf_counter() - t0) * 1000
    return {"content": content, "latency_ms": round(latency, 1)}
