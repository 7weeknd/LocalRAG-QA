"""RAG 主流程编排（LangChain 版）。

一次问答的完整链路：
1. 建索引：LangChain 加载器读文档 → 递归切块 → 向量化 → 存入向量库
2. 检索：向量库相似度检索（InMemoryVectorStore，无需额外装 FAISS）
3. 相关性判定：最高相似度 / 问题词覆盖率低于阈值 → 判定「知识库外」拒答（防幻觉）
4. 组装 Prompt：系统提示词（规则）+ 检索片段 + 用户问题
5. 生成：用 LCEL 把「Prompt | 模型 | 输出解析器」串成一条链（| 就是管道）
6. 返回：答案 + 是否命中 + 引用来源 + 耗时

与 main 分支的区别：检索与生成全部换成 LangChain 组件，不再手写 TF-IDF / BM25 / RRF。
对外契约保持不变（build() 返回片段数、ask() 返回同样六个字段），所以 routes / schemas / 前端都不用改。
"""
from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from .config import Config, load_config
from .loaders import build_documents

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

# 用户提示词模板：{context} / {question} 留空位，由 LCEL 在调用时填充
USER_TEMPLATE = "【知识库片段】\n{context}\n\n【用户问题】\n{question}"


def _is_ollama(base_url: str) -> bool:
    """本地 Ollama 与云端 OpenAI 兼容服务要用不同的客户端类。"""
    return any(k in base_url for k in (":11434", "localhost", "127.0.0.1"))


def _char_bigrams(text: str) -> set[str]:
    """把文本切成「字符二元组」集合，用来算问题词覆盖率。

    中文不依赖分词库也能得到合理的「词」：相邻两字为一个单元。
    这是计算覆盖率用的，不参与检索，所以不影响 LangChain 的检索链路。
    """
    cleaned = "".join(ch for ch in text if ch.strip())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


# 进程级单例：索引只需构建一次，app 入口与各个路由共享同一个实例
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(load_config())
    return _pipeline


class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.documents = []          # 切好的 LangChain Document 片段
        self.vectorstore = None      # 向量库
        self.chain = None            # LCEL 生成链（未配置大模型时为 None）

        base = (config.llm_base_url or "http://localhost:11434").rstrip("/")
        self.llm_model_name = config.llm_model
        self.llm_enabled = bool(config.llm_model)
        self.llm: BaseChatModel | None = None

        if self.llm_enabled:
            self.llm = self._build_llm(config, base)
            self.chain = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)]
            ) | self.llm | StrOutputParser()

        # 向量模型固定走 Ollama（本地私有化是项目的核心前提）
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model, base_url=base.replace("/v1", "")
        )

    @staticmethod
    def _build_llm(config: Config, base: str) -> BaseChatModel:
        """本地走 ChatOllama，云端走 ChatOpenAI —— 两者都是 LangChain 的聊天模型接口。"""
        if _is_ollama(base):
            return ChatOllama(
                model=config.llm_model,
                base_url=base,
                temperature=0.2,   # 问答要稳定，低温
                reasoning=False,   # 关键：关掉思考模式，否则 qwen3 等模型会返回空内容
            )
        return ChatOpenAI(
            model=config.llm_model,
            base_url=base,
            api_key=config.llm_api_key,
            temperature=0.2,
        )

    # ---------- 1. 建索引 ----------
    def build(self, data_dir: str | None = None) -> int:
        data_dir = data_dir or self.config.data_dir
        t0 = time.perf_counter()
        self.documents = build_documents(
            data_dir, chunk_size=self.config.chunk_size, overlap=self.config.overlap
        )
        # from_documents 会调用向量模型把每个片段编码成向量，再存进向量库
        self.vectorstore = InMemoryVectorStore.from_documents(self.documents, self.embeddings)
        self._build_ms = (time.perf_counter() - t0) * 1000
        return len(self.documents)

    # ---------- 2. 检索（向量相似度） ----------
    def retrieve(self, question: str, top_k: int | None = None) -> tuple[list[dict], float]:
        k = top_k or self.config.top_k
        pairs = self.vectorstore.similarity_search_with_score(question, k=k)
        results = [
            {
                "title": doc.metadata.get("title", "未命名"),
                "text": doc.page_content,
                "score": round(float(score), 4),
            }
            for doc, score in pairs
        ]
        max_score = max((r["score"] for r in results), default=0.0)
        return results, max_score

    # ---------- 相关性判定（防幻觉） ----------
    @staticmethod
    def _coverage(question: str, results: list[dict]) -> float:
        q_terms = _char_bigrams(question)
        if not q_terms:
            return 0.0
        combined = "".join(r["text"] for r in results)
        hit = sum(1 for t in q_terms if t in combined)
        return hit / len(q_terms)

    # ---------- 主入口 ----------
    def ask(self, question: str, top_k: int | None = None) -> dict:
        start = time.perf_counter()
        results, max_score = self.retrieve(question, top_k)

        if not results:
            delay = (time.perf_counter() - start) * 1000
            return self._result(FALLBACK_ANSWER, False, [], 0.0, 0.0, delay)

        coverage = self._coverage(question, results)
        relevant = max_score >= self.config.min_score or coverage >= self.config.min_coverage

        # 取前 final_k 个片段组装 Prompt
        top_results = results[: self.config.final_k]
        contexts = [f"[{r['title']}] {r['text']}" for r in top_results]

        if not relevant:
            answer = FALLBACK_ANSWER
        elif self.chain is not None:
            try:
                answer = self._generate(contexts, question)
            except Exception as exc:  # 大模型调用失败 → 抽取式兜底
                answer = self._extractive(
                    top_results, f"(大模型调用失败，已回退抽取式回答：{exc})"
                )
        else:
            answer = self._extractive(top_results)

        delay = (time.perf_counter() - start) * 1000
        citations = [
            {"title": r["title"], "score": r["score"], "snippet": r["text"][:120]}
            for r in top_results
        ]
        return self._result(answer, relevant, citations, max_score, coverage, delay)

    def _generate(self, contexts: list[str], question: str) -> str:
        """跑 LCEL 链：模板填变量 → 模型生成 → 解析成纯文本。"""
        ctx = "\n\n".join(f"【片段 {i + 1}】\n{c}" for i, c in enumerate(contexts))
        return self.chain.invoke({"context": ctx, "question": question}).strip()

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
