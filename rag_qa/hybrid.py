"""混合检索融合模块（RRF）。

为什么需要融合？
- 向量检索（稠密）擅长「语义相近」，关键词检索（BM25，稀疏）擅长「精确命中」。
- 单一检索器容易漏召回；把两者的排序结果融合，能取长补短。

RRF（Reciprocal Rank Fusion，倒数排名融合）：
    score(d) = Σ 1 / (k + rank(d))
- 不依赖各检索器的分数量纲（分数可能是相似度、也可能是 BM25 分值），
  只看「排名」，简单又鲁棒，是业界常用做法。
"""


def rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """对多个已排名的 (文档索引, 分数) 列表做 RRF 融合。

    ranked_lists 里每个子列表已是降序（排名越靠前越好）。
    返回按融合分降序的 (文档索引, 融合分) 列表，最多 top_k 个。
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(fused.items(), key=lambda x: -x[1])
    return ordered[:top_k]