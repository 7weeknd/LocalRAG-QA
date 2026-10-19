"""LocalRAG-QA · 本地私有化 RAG 知识库问答系统（LangChain 版）

一个可本地跑通、每一步都能讲清原理的全栈检索增强生成（RAG）项目。

检索链路（本分支用 LangChain 组件实现）：
- config     : 配置项（环境变量覆盖；含向量模型名与实测校准后的防幻觉阈值）
- loaders    : 文档加载（LangChain Loader：TextLoader / PyPDFLoader / 自定义 ExcelLoader）
                + 递归切块（RecursiveCharacterTextSplitter）
- pipeline   : RAG 主流程编排（向量化入库 → 相似度检索 → 相关性判定 → LCEL 链生成）

全栈部分（与检索层解耦，main 与 langchain 两个分支共用，未改动）：
- database   : SQLAlchemy 引擎 / 会话 / 启动时自动建库建表
- models     : ORM 模型：User / ChatHistory
- schemas    : Pydantic 入参出参
- security   : bcrypt 密码哈希 + JWT 签发校验
- routes     : 注册 / 登录 / 问答落库 / 历史查询
- cli        : 命令行问答入口
- app        : FastAPI 服务入口

> main 分支的检索是手写实现（TF-IDF / BM25 / RRF），本分支改为 LangChain 组件并做了实测对比。
"""

__version__ = "0.1.0"
