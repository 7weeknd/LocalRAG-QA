"""命令行问答入口。

用法（在项目根目录 LocalRAG-QA/ 下）：
    python -m rag_qa.cli
交互式输入问题，输入 exit / q 退出。
"""
from .config import load_config
from .pipeline import RAGPipeline


def main():
    import sys
    try:
        # 兼容 GBK 等非 UTF-8 终端：遇到无法编码的字符用 ? 替换，避免直接崩溃
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    config = load_config()
    pipeline = RAGPipeline(config)
    n = pipeline.build()
    print(f"[OK] 索引构建完成：{n} 个知识片段，耗时 {pipeline._build_ms:.1f} ms")
    print(f"   大模型：{'已接入 ' + config.llm_model if pipeline.llm.enabled else '未配置（抽取式演示）'}")
    print("输入问题开始问答（输入 exit 退出）\n")

    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q", "退出"):
            break
        if not q:
            continue

        r = pipeline.ask(q)
        print(f"答：{r['answer']}")
        print(f"命中：{r['hit']} | 引用：{[c['title'] for c in r['citations']]} | 耗时 {r['latency_ms']} ms\n")


if __name__ == "__main__":
    main()