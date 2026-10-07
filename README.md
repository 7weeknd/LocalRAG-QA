# LocalRAG-QA

**本地私有化 RAG 知识库问答系统** · 检索算法全部手写（TF-IDF / BM25 / RRF）

基于本地大模型构建的检索增强生成（RAG）问答系统，面向电商客服知识库场景，
对「退换货 / 会员 / 物流 / 发票 / 优惠券 / FAQ」六类知识做问答。
**数据全流程不出本机**，TF-IDF、BM25、RRF 等检索算法均为手写实现，不依赖 LangChain 等框架。

---

## 一、系统架构

```
用户提问
   │
   ▼
┌─────────────┐   ┌──────────────────────────────┐
│  中文分词     │──▶│  检索层：                        │
│  (jieba)    │   │  · TF-IDF 向量检索（语义）        │──▶ RRF 融合 ──▶ 相关性判定
└─────────────┘   │  · BM25 关键词检索（精确）        │
                  └──────────────────────────────┘          │
                                            相关            │ 不相关
                                              ▼            ▼
                          ┌──────────────┐          ┌────────────┐
                          │ 组装 Prompt   │          │ 兜底拒答    │
                          │ 调用大模型     │          │ (转人工)    │
                          └──────┬───────┘          └────────────┘
                                 ▼
                          答案 + 引用来源 + 耗时
```

一次问答经历 6 步：**分词 → 切片检索 → 混合融合 → 相关性判定（防幻觉）→ 组装 Prompt → 生成**。

---

## 二、核心原理

### 1. 中文分词 `tokenizer.py`
- 中文无空格，检索前必须切词。
- 方案：jieba 词典分词；无依赖时降级为**字符二元组（n-gram）**，对未登录词更鲁棒。
- 同时做**停用词过滤**（去掉"的/了/吗/怎么"等功能词与标点），避免无信息量词干扰相似度。

### 2. 文档分块 `chunker.py`
- 大模型上下文有限，必须切片；粒度越细越精准，但太细会「断章取义」。
- 两参数：`chunk_size`（块大小）、`overlap`（重叠，防关键句卡在边界漏召回）。

### 3. TF-IDF 向量化 `embeddings.py`
- `TF-IDF = 词频 × 逆文档频率`，把文档表示成稀疏向量，用**余弦相似度**做软匹配。
- 相比字符串匹配，能容忍语序/部分词差异。

### 4. BM25 关键词检索 `bm25.py`
- 带**词频饱和 + 长度归一化**的经典排序算法（Elasticsearch 同款思路）。
- 优势：精确词命中（订单号、专有名词）。

### 5. 混合检索 + RRF `hybrid.py`
- 向量（语义）+ 关键词（精确）**取长补短**。
- RRF 只看排名不看分数量纲，公式 `score = Σ 1/(k + rank)`，鲁棒通用。

### 6. 防幻觉（关键设计）`pipeline.py`
- 用「最高相似度」与「问题词覆盖率」双阈值判定相关性，**检索不到就拒答**，而不是让大模型硬编。

### 7. 工程化 `app.py`
- 索引**一次性构建到内存**，请求阶段只做检索（性能调优思路）；
- FastAPI + Pydantic 服务化，返回结构化 JSON。

---

## 三、快速开始

> 环境要求：Python 3.10+（本机若还没装，先去 https://www.python.org 装一个，勾选 "Add to PATH"）

### 1. 安装依赖
```bash
cd LocalRAG-QA
pip install -r requirements.txt
```

### 2. 跑通演示（无需大模型 Key，抽取式回答）
```bash
python -m rag_qa.cli
```
输入问题试试：
```
你：七天无理由退货运费谁出？
你：黑金会员有什么权益？
你：怎么开电子发票？
你：今天天气怎么样？     ← 知识库外问题，验证「拒答兜底」
```

### 3. 接入真实大模型（可选）
```bash
# Windows PowerShell
$env:LLM_BASE_URL="https://api.deepseek.com/v1"
$env:LLM_API_KEY="sk-xxxx"
$env:LLM_MODEL="deepseek-chat"
python -m rag_qa.cli
```

### 4. 启动 API 服务
```bash
uvicorn rag_qa.app:app --host 0.0.0.0 --port 8000
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"七天无理由退货运费谁出？"}'
```

### 5. 跑测试
```bash
python tests/test_retrieval.py
```

---

## 四、目录结构
```
LocalRAG-QA/
├── rag_qa/
│   ├── config.py      # 配置（环境变量覆盖）
│   ├── tokenizer.py   # 中文分词
│   ├── chunker.py     # 文档加载 + 分块
│   ├── embeddings.py  # TF-IDF 向量化
│   ├── bm25.py        # BM25 关键词检索
│   ├── hybrid.py      # RRF 混合融合
│   ├── llm.py         # 大模型调用（OpenAI 兼容）
│   ├── pipeline.py    # RAG 主流程编排
│   ├── cli.py         # 命令行问答
│   └── app.py         # FastAPI 服务
├── data/docs/         # 知识库（6 篇电商客服文档）
├── tests/             # 冒烟测试
├── requirements.txt
└── README.md
```

---

## 五、后续可扩展方向

1. **稠密向量**：接入 sentence-transformers，把 TF-IDF 稀疏向量换成语义向量，效果显著提升。
2. **重排序 Rerank**：混合检索后再加一层 cross-encoder 精排，进一步压准确率。
3. **评估指标**：构造测试集，算 Recall@k / MRR，量化检索质量。
4. **流式输出 + Docker 部署**：SSE 流式返回、容器化，贴近生产。
5. **持久化索引**：把 TF-IDF 矩阵 / 向量库落盘，避免每次启动重建。
