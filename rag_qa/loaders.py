"""文档加载与切块（LangChain 版）。

与 main 分支的区别：这里不再手写文件解析和切片，全部交给 LangChain 的组件。

链路：文档 → LangChain Document 对象 → RecursiveCharacterTextSplitter → 片段列表

为什么用 RecursiveCharacterTextSplitter：
它是「按优先级递归切」——先试着用段落分（\\n\\n），切不动再退到换行、句号、空格，
好处是尽量不把一句话或一个段落从中间切断，比固定字符窗口切更符合语义边界。

支持的格式与 main 分支保持一致：md / txt / pdf / xlsx。
md / txt 用 LangChain 自带的 TextLoader，pdf 用 PyPDFLoader，
Excel 没有开箱即用的轻量加载器（官方的 UnstructuredExcelLoader 要装一大堆依赖），
所以自定义一个 ExcelLoader，继承 LangChain 的 BaseLoader —— 自定义 Loader 是这个框架的标准扩展点。
"""
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 各格式对应的后缀（与 main 分支一致）
TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}
SHEET_EXTS = {".xlsx", ".xlsm"}


class ExcelLoader(BaseLoader):
    """把 Excel 每个工作表转成一页文本的 LangChain Loader。

    继承 BaseLoader 只需要实现 lazy_load()：产出 Document 对象。
    """

    def __init__(self, file_path: str | Path):
        self.file_path = str(file_path)
        self.source = Path(file_path).stem

    def lazy_load(self):
        try:
            import openpyxl
        except ImportError:
            raise RuntimeError(
                f"读取 Excel 需要 openpyxl，请先安装：pip install openpyxl（文件：{self.source}）"
            )
        wb = openpyxl.load_workbook(self.file_path, data_only=True)
        lines: list[str] = []
        for ws in wb.worksheets:
            lines.append(f"【工作表：{ws.title}】")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
        # 标题取文件名（Excel 没有 # 标题行可提取）
        yield Document(page_content="\n".join(lines), metadata={"title": self.source})


def _pick_title(text: str, fallback: str) -> str:
    """标题优先取首个 `#` 行，否则回退到文件名（与 main 分支逻辑一致）。"""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def _load_text(path: Path) -> list[Document]:
    """md / txt：用 LangChain 的 TextLoader 读，再补上标题元数据。"""
    docs = TextLoader(str(path), encoding="utf-8").load()
    for d in docs:
        d.metadata["title"] = _pick_title(d.page_content, path.stem)
    return docs


def load_documents(data_dir: str) -> list[Document]:
    """遍历知识库目录，把每种格式都加载成 LangChain Document。"""
    root = Path(data_dir)
    if not root.exists():
        return []

    docs: list[Document] = []
    for ext in TEXT_EXTS | PDF_EXTS | SHEET_EXTS:
        for path in sorted(root.glob(f"*{ext}")):
            if path.suffix.lower() in TEXT_EXTS:
                docs.extend(_load_text(path))
            elif path.suffix.lower() in PDF_EXTS:
                docs.extend(PyPDFLoader(str(path)).load())
            else:
                docs.extend(ExcelLoader(path).load())
    return docs


def build_documents(
    data_dir: str, chunk_size: int = 300, overlap: int = 50
) -> list[Document]:
    """加载 + 切块，返回切好的片段列表（等价于 main 分支的 build_chunks）。"""
    docs = load_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # 中文场景：优先按段落 / 换行切，再退到句读
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    return splitter.split_documents(docs)
