"""检索正确性的冒烟测试。

运行（在项目根目录 LocalRAG-QA/ 下）：
    python -m pytest tests/test_retrieval.py -v
或直接：
    python tests/test_retrieval.py
"""
import sys
from pathlib import Path

# 保证能 import 到 rag_qa 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.config import load_config  # noqa: E402
from rag_qa.pipeline import RAGPipeline  # noqa: E402


def _pipeline():
    config = load_config()
    config.data_dir = str(Path(__file__).resolve().parent.parent / "data" / "docs")
    pipe = RAGPipeline(config)
    pipe.build()
    return pipe


def test_build():
    pipe = _pipeline()
    assert len(pipe.chunks) > 0


def test_retrieve_refund():
    pipe = _pipeline()
    r = pipe.ask("七天无理由退货运费由谁承担？")
    assert r["hit"] is True
    joined = "".join(c["title"] for c in r["citations"])
    assert "退换货" in joined


def test_fallback_unknown():
    pipe = _pipeline()
    r = pipe.ask("今天北京的天气怎么样？")
    assert r["hit"] is False


if __name__ == "__main__":
    test_build()
    test_retrieve_refund()
    test_fallback_unknown()
    print("全部测试通过")