"""TF-IDF 向量化模块。

这是最经典的「搜索算法」之一，讲原理必考题：
- TF（词频）：一个词在某文档中出现的次数越多，越能代表该文档。
- IDF（逆文档频率）：一个词在「多少篇文档」中出现过，越普遍越不重要。
- TF-IDF = TF * IDF，得到一个稀疏向量；两向量夹角余弦 = 相似度。

为什么不用纯字符串匹配？
- 字符串匹配只能「完全一致」，无法处理近义、同义、语序变化。
- 向量化后可用余弦相似度做「软匹配」，容忍部分词不同。

这里用 numpy 手工实现（不依赖 sklearn），保证你能讲清每一步公式。
"""
import math
from collections import Counter

import numpy as np


class TfidfEmbedder:
    """对语料拟合 TF-IDF 矩阵，并支持对查询向量化 + 余弦检索。"""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.matrix: np.ndarray | None = None  # (n_docs, vocab)，已 L2 归一化

    def fit(self, token_lists: list[list[str]]) -> "TfidfEmbedder":
        n = len(token_lists)

        # 统计每个词出现在多少篇文档里（document frequency）
        df = Counter()
        for toks in token_lists:
            for t in set(toks):
                df[t] += 1

        self.vocab = {t: i for i, t in enumerate(sorted(df))}

        # 平滑 IDF：避免 df=0 时除零，sklearn 同款公式
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((n + 1) / (df[t] + 1)) + 1.0

        # 逐文档构造 TF-IDF 向量并做 L2 归一化（便于用余弦相似度）
        self.matrix = np.zeros((n, len(self.vocab)))
        for r, toks in enumerate(token_lists):
            tf = Counter(toks)
            for t, c in tf.items():
                if t in self.vocab:
                    self.matrix[r, self.vocab[t]] = c
            self.matrix[r] = self.matrix[r] * self.idf
            norm = np.linalg.norm(self.matrix[r])
            if norm > 0:
                self.matrix[r] /= norm

        return self

    def _encode(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        tf = Counter(tokens)
        for t, c in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = c
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        """返回按余弦相似度降序的 (文档索引, 相似度) 列表。"""
        qv = self._encode(query_tokens)
        scores = self.matrix @ qv
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]