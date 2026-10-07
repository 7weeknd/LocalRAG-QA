"""RAG 主流程编排模块。

一次问答的完整链路：
1. 建索引：加载文档 → 分块 → 分词 → 拟合 TF-IDF 与 BM25
2. 检索：向量（TF-IDF 余弦） + 关键词（BM25） → RRF 融合
3. 相关性判定：相似度 或 覆盖率 低于阈值 → 判定「知识库外」拒答（防幻觉）
4. 组装 Prompt：系统提示词（规则）+ 检索片段 + 用户问题
5. 生成：有 Key 走大模型，无 Key 抽取式兜底
6. 返回：答案 + 是否命中 + 引用来源 + 耗时
"""
from __future__ import annotations

import time

from .bm25 import BM25
from .chunker import build_chunks
from .config import Config
from .embeddings import TfidfEmbedder
from .hybrid import rrf_fuse
from .llm import LLMClient
from .tokenizer import tokenize

# 生成回答的系统提示词（静态规则）
SYSTEM_PROMPT = (
    "你是基于知识库的智能问答助手。请严格依据下面提供的【知识库片段】回答用户问题，"
    "禁止编造。涉及数字（天数、金额、比例、时效）必须与片段完全一致。"
    "若片段无法回答用户问题，请直接输出：'抱歉，根据现有资料暂未查询到相关信息。'"
    "回答要求：先给结论，再分条说明，简洁专业，不超过 200 字。"
)

# 知识库外问题的兜底话术
FALLBACK_ANSWER = (
    "抱歉，根据现有资料暂未查询到相关信息，"
    "建议您换个问法，或联系人工客服进一步确认。"
)


def _build_user_prompt(contexts: list[str], question: str) -> str:
    ctx = "\n\n".join(f"【片段 {i + 1}】\n{c}" for i, c in enumerate(contexts))
    return f"{ctx}\n\n【用户问题】\n{question}"


class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.chunks = []                 # 所有知识片段
        self.token_lists: list[list[str]] = []
        self.embedder = TfidfEmbedder()
        self.bm25 = BM25()
        self.llm = LLMClient(config.llm_base_url, config.llm_api_key, config.llm_model)

    # ---------- 1. 建索引 ----------
    def build(self, data_dir: str | None = None) -> int:
        data_dir = data_dir or self.config.data_dir
        t0 = time.perf_counter()
        self.chunks = build_chunks(
            data_dir,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
        )
        self.token_lists = [tokenize(c.text, self.config.use_jieba) for c in self.chunks]
        self.embedder.fit(self.token_lists)
        self.bm25.fit(self.token_lists)
        self._build_ms = (time.perf_counter() - t0) * 1000
        return len(self.chunks)

    # ---------- 2. 检索 ----------
    def retrieve(self, question: str, top_k: int | None = None):
        top_k = top_k or self.config.top_k
        qt = tokenize(question, self.config.use_jieba)
        if not qt:  # 分词/去停用词后无有效词，直接判知识库外
            return [], 0.0, qt

        dense = self.embedder.search(qt, top_k)    # 向量：语义相似
        sparse = self.bm25.search(qt, top_k)       # 关键词：精确命中
        fusion = rrf_fuse([dense, sparse], k=self.config.rrf_k, top_k=top_k)

        max_cos = dense[0][1] if dense else 0.0
        results = [
            {
                "idx": idx,
                "title": self.chunks[idx].title,
                "text": self.chunks[idx].text,
                "score": round(score, 4),
            }
            for idx, score in fusion
        ]
        return results, max_cos, qt

    # ---------- 相关性判定（防幻觉） ----------
    @staticmethod
    def _coverage(query_tokens: list[str], results: list[dict]) -> float:
        if not query_tokens:
            return 0.0
        combined = "".join(r["text"] for r in results)
        hit = sum(1 for t in query_tokens if t in combined)
        return hit / len(query_tokens)

    # ---------- 主入口 ----------
    def ask(self, question: str, top_k: int | None = None) -> dict:
        start = time.perf_counter()
        results, max_cos, qt = self.retrieve(question, top_k)

        if not results:
            delay = (time.perf_counter() - start) * 1000
            return self._result(FALLBACK_ANSWER, False, [], max_cos, 0.0, delay)

        coverage = self._coverage(qt, results)
        relevant = max_cos >= self.config.min_score or coverage >= self.config.min_coverage

        # 取融合后前 final_k 个片段组装 Prompt
        top_results = results[: self.config.final_k]
        contexts = [f"[{r['title']}] {r['text']}" for r in top_results]

        answer = ""
        if not relevant:
            answer = FALLBACK_ANSWER
        elif self.llm.enabled:
            try:
                answer = self.llm.generate(SYSTEM_PROMPT, _build_user_prompt(contexts, question))
            except Exception as exc:  # 大模型调用失败 → 抽取式兜底
                answer = self._extractive(top_results, f"(大模型调用失败，已回退抽取式回答：{exc})")
        else:
            answer = self._extractive(top_results)

        delay = (time.perf_counter() - start) * 1000
        citations = [
            {"title": r["title"], "score": r["score"], "snippet": r["text"][:120]}
            for r in top_results
        ]
        return self._result(answer, relevant, citations, max_cos, coverage, delay)

    @staticmethod
    def _extractive(top_results: list[dict], note: str = "") -> str:
        """抽取式兜底：直接返回最相关片段（演示/离线模式）。"""
        if not top_results:
            return FALLBACK_ANSWER
        text = top_results[0]["text"].strip()
        return (note + "\n" if note else "") + text

    @staticmethod
    def _result(answer, hit, citations, max_score, coverage, latency_ms) -> dict:
        return {
            "answer": answer,
            "hit": hit,
            "citations": citations,
            "max_score": round(max_score, 4),
            "coverage": round(coverage, 4),
            "latency_ms": round(latency_ms, 1),
        }