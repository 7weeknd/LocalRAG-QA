"""BM25 关键词检索模块（稀疏检索）。

BM25 是搜索引擎（如 Elasticsearch 默认相关度）里的经典排序算法，
相比朴素 TF-IDF 多了「词频饱和」和「长度归一化」两个修正：

score(q, d) = Σ IDF(t) * [ f(t,d)*(k1+1) ] / [ f(t,d) + k1*(1 - b + b*|d|/avgdl) ]

- k1：控制词频饱和速度，出现很多次不会无限加分；
- b ：控制文档长度惩罚力度，长文档不会天然占便宜。

在 RAG 里，关键词检索（BM25）与向量检索互补：
向量擅长语义相近，BM25 擅长精确词命中（如订单号、专有名词）。
"""
import math
from collections import Counter


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}

    def fit(self, token_lists: list[list[str]]) -> "BM25":
        self.doc_tokens = token_lists
        self.doc_len = [len(t) for t in token_lists]
        self.avgdl = sum(self.doc_len) / max(1, len(token_lists))

        for toks in token_lists:
            for t in set(toks):
                self.df[t] += 1

        n = len(token_lists)
        for t, c in self.df.items():
            # 带平滑的 IDF
            self.idf[t] = math.log((n - c + 0.5) / (c + 0.5) + 1.0)
        return self

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = Counter(self.doc_tokens[doc_idx])
        dl = self.doc_len[doc_idx]
        score = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self.idf.get(t, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        """返回按 BM25 分数降序的 (文档索引, 分数) 列表。"""
        scored = [(self._score(query_tokens, i), i) for i in range(len(self.doc_tokens))]
        scored.sort(key=lambda x: -x[0])
        return [(i, s) for s, i in scored[:top_k] if s > 0]