"""知识库加载与切块冒烟测试。

跑法（项目根目录下）：python tests/test_retrieval.py

只验证「加载 → LangChain 切块」这条链路，不调用大模型和向量模型，
所以不需要启动 Ollama 也能跑，适合当提交前的快速自检。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.loaders import build_documents, load_documents  # noqa: E402

DATA_DIR = "data/docs"


def main():
    docs = load_documents(DATA_DIR)
    print(f"[1] 加载文档：{len(docs)} 篇")
    assert docs, "没加载到任何文档，检查 data/docs 目录"

    chunks = build_documents(DATA_DIR, chunk_size=300, overlap=50)
    print(f"[2] 切块后片段：{len(chunks)} 个")
    assert chunks, "切块结果为空"
    assert len(chunks) >= len(docs), "片段数不应少于文档数"

    titles = sorted({c.metadata.get("title", "") for c in chunks})
    print(f"[3] 覆盖标题 {len(titles)} 个：{titles}")
    assert all(titles), "存在没有标题的片段（元数据丢了）"

    longest = max(len(c.page_content) for c in chunks)
    print(f"[4] 最长片段 {longest} 字符（chunk_size=300）")
    assert longest <= 340, "切块没有生效，片段超长"

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
