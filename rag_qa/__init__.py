"""LocalRAG-QA · 本地私有化 RAG 知识库问答系统

一个可本地跑通、每一步都能讲清原理的全栈检索增强生成（RAG）项目。

模块划分：
- config     : 配置项（用环境变量覆盖）
- tokenizer  : 中文分词（jieba，无依赖时降级为字符二元组）
- chunker    : 文档加载 + 分块（固定窗口 + 重叠）
- embeddings : TF-IDF 向量化（稀疏向量 + 余弦相似度）
- bm25       : BM25 关键词检索（自实现）
- hybrid     : 混合检索融合（RRF，Reciprocal Rank Fusion）
- llm        : 大模型调用（OpenAI 兼容接口），无 Key 时抽取式兜底
- pipeline   : RAG 主流程编排（检索 → 相关性判定 → 组装 Prompt → 生成）
- cli        : 命令行问答入口
- app        : FastAPI 服务入口
"""

__version__ = "0.1.0"