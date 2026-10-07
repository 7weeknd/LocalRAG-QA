"""中文分词模块。

为什么检索前要先分词？
- 中文不像英文有天然空格分隔，需要把连续文本切成「词」才能建立倒排/向量索引。
- 方案一：jieba（词典+概率），分词更符合语义。
- 方案二：字符二元组（n-gram），无需依赖、对未登录词更鲁棒，适合做兜底。

这里做成：优先 jieba，`import` 失败时自动降级为字符二元组，
保证「开箱即跑」，同时保留工程上更专业的分词路径。
"""
from __future__ import annotations

from typing import List

_JIEBA_AVAILABLE: bool | None = None

# 常见中文停用词：检索时过滤掉这些「无信息量」的功能词，
# 避免它干扰相似度计算（例如"的/了/吗"几乎出现在所有文档里）。
STOPWORDS = {
    "的", "了", "吗", "呢", "啊", "吧", "呀", "哦", "嗯", "么",
    "是", "在", "有", "和", "与", "或", "及", "就", "都", "还",
    "要", "会", "能", "把", "被", "让", "给", "对", "从", "到",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "一个", "什么", "怎么", "怎样",
    "怎么样", "为啥", "为什么", "请问", "一下", "谁", "由", "哪些",
    "啥", "哪里", "哪儿", "如何", "可以", "应该", "需要", "是否",
}


def _filter(tokens: List[str]) -> List[str]:
    """去掉纯标点 / 空白 token，以及停用词。"""
    out: List[str] = []
    for t in tokens:
        if not t or not any(ch.isalnum() for ch in t):  # 纯标点/空白
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


def _has_jieba() -> bool:
    global _JIEBA_AVAILABLE
    if _JIEBA_AVAILABLE is None:
        try:
            import jieba  # noqa: F401
            _JIEBA_AVAILABLE = True
        except ImportError:
            _JIEBA_AVAILABLE = False
    return _JIEBA_AVAILABLE


def _jieba_cut(text: str) -> List[str]:
    import jieba
    return [w for w in jieba.lcut(text) if w.strip()]


def _char_bigram(text: str) -> List[str]:
    """字符二元组：单字 + 相邻双字，作为无 jieba 时的降级方案。"""
    s = "".join(text.split())
    tokens: List[str] = []
    for i, ch in enumerate(s):
        tokens.append(ch)
        if i + 1 < len(s):
            tokens.append(s[i:i + 2])
    return tokens


def tokenize(text: str, use_jieba: bool = True) -> List[str]:
    raw = _jieba_cut(text) if (use_jieba and _has_jieba()) else _char_bigram(text)
    return _filter(raw)