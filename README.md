# LocalRAG-QA

**本地私有化 RAG 知识库问答系统** · 检索与生成链路使用 **LangChain** 构建

> 早期版本的检索链路（TF-IDF / BM25 / RRF）是**手写实现**的，现已整体替换为 LangChain 组件。
> 手写版完整保留在 git 历史里，想看两版差异：`git diff f3dadc2 main`。

## 两个版本的区别

| | 早期手写版（历史提交 `f3dadc2`） | 当前版本（LangChain） |
|---|---|---|
| 文档加载 / 切块 | 手写 `chunker.py` | LangChain 加载器 + `RecursiveCharacterTextSplitter` |
| 检索 | 手写 TF-IDF + BM25 + RRF 融合 | `OllamaEmbeddings` + `InMemoryVectorStore` 向量检索 |
| 生成 | 手写 requests 调 `/api/chat` | LCEL 链：`Prompt \| ChatOllama \| StrOutputParser` |
| 中文分词 | jieba + n-gram 降级 | 不需要（交给向量模型） |
| 依赖 | 少（numpy / requests） | 多（langchain-core / langchain-ollama / langchain-community） |
| 手写算法 | TF-IDF、BM25、RRF 全部手写 | 无，全部用框架 |
| 对外接口 | 完全一致（路由 / 数据库 / 前端都不用改） | 同左 |

---

## 一、系统架构（LangChain 版）

```
用户提问
   │
   ▼
┌──────────────────────────────────────────────┐
│ 建索引（启动时一次）                            │
│  Loader（md/txt/pdf/xlsx）→ Document 对象      │
│         ↓                                     │
│  RecursiveCharacterTextSplitter 递归切块        │
│         ↓                                     │
│  OllamaEmbeddings 向量化 → InMemoryVectorStore │
└──────────────────────┬───────────────────────┘
                       ▼
            similarity_search_with_score(question, k)
                       │
                       ▼
              相关性判定（防幻觉）→ 不相关则拒答
                       │ 相关
                       ▼
   LCEL 链：ChatPromptTemplate | ChatOllama | StrOutputParser
                       │
                       ▼
              答案 + 引用来源 + 耗时
```

**LCEL 是什么**：LangChain Expression Language，用 `|` 把组件串成管道。
`Prompt | LLM | Parser` 读起来就是「填模板 → 调模型 → 解析成纯文本」，
每个组件都是 `Runnable`，可以单独替换、单独测试。

---

## 二、核心实现

### 1. 文档加载 `loaders.py`
- **md / txt**：LangChain 自带的 `TextLoader`
- **pdf**：`PyPDFLoader`（依赖 pypdf）
- **Excel**：官方 `UnstructuredExcelLoader` 要装一堆依赖，所以**自定义 `ExcelLoader`**——
  继承 `BaseLoader` 实现 `lazy_load()`，产出一个 `Document`。
  这是 LangChain 的标准扩展点：框架没覆盖的格式，自己写一个 Loader 即可接入整条链。
- 所有格式最后都统一成 LangChain 的 `Document`（`page_content` + `metadata`），
  下游切块器、向量库只认这一种数据结构，所以加格式不影响后面任何环节。

### 2. 切块 `RecursiveCharacterTextSplitter`
按「优先级递归」切：先试段落（`\n\n`），切不动退到换行，再退到句号、分号。
好处是尽量不在句子中间切断。中文场景需要在 `separators` 里显式加上「。！？；」。

### 3. 向量化与向量库 `pipeline.py`
- `OllamaEmbeddings(model="nomic-embed-text")`：本地向量模型，数据不出本机
- `InMemoryVectorStore`：langchain-core 自带，**不用额外装 FAISS**
- 检索用 `similarity_search_with_score()`，拿到「文本 + 相似度分数」用于后面的阈值判定

### 4. 生成：LCEL 链
```python
chain = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),      # 静态规则
    ("human", USER_TEMPLATE),       # {context} / {question} 占位
]) | ChatOllama(model=..., reasoning=False) | StrOutputParser()
```

`reasoning=False` 是**关键**：qwen 系列的思考版模型会把思考内容塞满输出预算，
导致 `content` 为空。新版 `ChatOllama` 支持用 `reasoning` 参数关掉思考模式。

---

## 三、防幻觉：阈值是实测校准出来的，不是拍脑袋写的

换成向量检索后，相似度的分布和 TF-IDF 完全不同，所以阈值必须重新标定。
实测数据（`nomic-embed-text` + 12 个片段）：

| 问题 | 最高相似度 | 问题词覆盖率 |
|------|-----------|-------------|
| 知识库内：七天无理由退货运费谁出？ | 0.6723 | 0.6364 |
| 知识库内：黑金会员有什么权益？ | 0.7393 | 0.4444 |
| 知识库内：怎么开电子发票？ | 0.6799 | 0.4286 |
| 知识库外：今天天气怎么样？ | 0.6578 | **0.00** |
| 知识库外：帮我写一段冒泡排序 | 0.7128 | **0.00** |
| 知识库外：推荐几家好吃的火锅店 | 0.6209 | **0.00** |

**结论（重要）**：两组相似度区间**完全重叠**——在短中文问句上，
向量相似度根本区分不出「相关」和「不相关」。而覆盖率两组分得干干净净（库内 ≥0.43，库外 0.00）。

所以最终策略：**以覆盖率为主判据**，相似度阈值定在实测上界之上只做兜底。
判定准确率 6/6（实测）。

> 这条「实测发现 embedding 相似度区分度不足，改用覆盖率主判」的经历，
> 比直接写「我用了双阈值」有价值得多——因为它是测量出来的，不是抄来的。

---

## 四、已知不足（如实记录）

1. **本地 0.8B 模型指令遵循能力弱**：实测「怎么开电子发票？」检索完全正确
   （答案就在 Top-1 片段里，已用诊断脚本确认），但模型仍输出兜底话术。
   这是**模型能力问题，不是检索问题**——换成更大的模型（改 `LLM_MODEL`）即可。
2. **覆盖率对同义改写不友好**：「怎么退货」能命中，「想退货可以吗」这类改写
   覆盖率会掉到阈值以下而被拒答。后续可用「LLM 判定相关性」或 query rewriting 改进。

---

## 五、快速开始

> 环境要求：Python 3.10+；MySQL（全栈版需要）；Ollama

```bash
# 1. 模型：对话模型 + 向量模型
ollama pull qwen3.5:0.8b
ollama pull nomic-embed-text        # 向量化用，约 274 MB

# 2. 依赖
cd LocalRAG-QA
pip install -r requirements.txt
```

启动：

```bash
python -m rag_qa.cli                 # 命令行版
uvicorn rag_qa.app:app --port 8010   # API + Web 版（或双击 start_web.bat）
```

`.env` / 环境变量与 main 分支一致，新增一个 `EMBEDDING_MODEL`（默认 `nomic-embed-text`）。

### 跑测试

```bash
python tests/test_retrieval.py       # 加载 + 切块冒烟测试，不依赖 Ollama
```

---

## 六、接口与前端：与 main 分支完全一致

`ask()` 的返回结构没变（`answer / hit / citations / max_score / coverage / latency_ms`），
所以 `routes.py`、`schemas.py`、`database.py`、`security.py`、`static/index.html` **一行都不用改**。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录，返回 JWT |
| POST | `/api/ask` | 问答，自动写入对话历史 |
| GET | `/api/chat/history` | 查自己的对话历史 |
| POST | `/ask` | 公开问答（不记录历史） |
| GET | `/health` | 服务状态 + 当前模型 |

---

## 七、目录结构

```
LocalRAG-QA/
├── rag_qa/
│   ├── config.py      # 配置（含 embedding_model / 校准后的阈值）
│   ├── loaders.py     # 【新】LangChain 加载器 + 递归切块
│   ├── pipeline.py    # 【重写】向量库 + 检索 + LCEL 生成链
│   ├── app.py         # FastAPI 入口
│   ├── database.py    # SQLAlchemy 引擎 / 会话 / 自动建库
│   ├── models.py      # ORM：User / ChatHistory
│   ├── schemas.py     # Pydantic 入参出参
│   ├── security.py    # bcrypt + JWT
│   └── routes.py      # 接口层
├── static/index.html  # 单页前端
├── data/docs/         # 知识库（6 篇电商客服文档）
├── tests/             # 冒烟测试
└── README.md
```

> 早期手写版里的 `tokenizer.py`、`embeddings.py`、`bm25.py`、`hybrid.py`、`chunker.py`、`llm.py`
> 已不在当前版本中（需要时从 git 历史 `f3dadc2` 取回）。

---

## 八、后续可扩展方向

1. **换更大的本地模型**（如 qwen3:7b）解决小模型拒答问题。
2. **Rerank**：`InMemoryVectorStore` 粗排后加一层 cross-encoder 精排。
3. **向量库落盘**：把 `InMemoryVectorStore` 换成 FAISS / Chroma，避免每次启动重建索引。
4. **评估指标**：构造测试集算 Recall@k / MRR，量化检索质量。
