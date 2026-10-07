"""文档加载与分块（Chunking）模块。

为什么要分块？
- 大模型上下文有限，不能一次塞入整本知识库；
- 检索粒度越细，命中的片段越精准、噪音越少。
两个关键参数：
- chunk_size：块越大信息越完整但越杂，块越小越精准但易「断章取义」。
- overlap：相邻块重叠部分，避免关键句恰好被切在边界导致漏召回。
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """一个知识片段：属于哪篇文档、标题、正文。"""
    chunk_id: int
    doc_id: str
    title: str
    text: str


# 支持的文件类型：文本（utf-8）、PDF、Excel（xlsx/xlsm）
TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}
SHEET_EXTS = {".xlsx", ".xlsm"}


def list_documents(data_dir: str):
    """列出知识库目录下所有支持的文档（md / txt / pdf / xlsx）。"""
    root = Path(data_dir)
    if not root.exists():
        return []
    files = []
    for ext in TEXT_EXTS | PDF_EXTS | SHEET_EXTS:
        files.extend(root.glob(f"*{ext}"))
    return sorted(files, key=lambda p: p.name)


def _pick_title(text: str, fallback: str) -> str:
    """标题优先取首个 `#` 行，否则回退到文件名。"""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def _load_text(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    return _pick_title(text, path.stem), text


def _load_pdf(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(f"读取 PDF 需要 pypdf，请先安装：pip install pypdf（文件：{path.name}）")
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    text = "\n\n".join(pages).strip()
    return path.stem, text


def _load_sheet(path: Path) -> tuple[str, str]:
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError(f"读取 Excel 需要 openpyxl，请先安装：pip install openpyxl（文件：{path.name}）")
    wb = openpyxl.load_workbook(path, data_only=True)
    lines = []
    for ws in wb.worksheets:
        lines.append(f"【工作表：{ws.title}】")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append(" | ".join(cells))
    return path.stem, "\n".join(lines)


def load_document(path) -> tuple[str, str]:
    """读取一篇文档，返回 (标题, 正文)。按后缀分发到对应解析器。"""
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTS:
        return _load_text(path)
    if suffix in PDF_EXTS:
        return _load_pdf(path)
    if suffix in SHEET_EXTS:
        return _load_sheet(path)
    return path.stem, ""


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """按字符窗口切分，相邻块重叠 overlap 个字符。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def build_chunks(data_dir: str, chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    """遍历知识库目录，把每篇文档切成片段，返回带自增 id 的片段列表。"""
    chunks: list[Chunk] = []
    for path in list_documents(data_dir):
        title, text = load_document(path)
        for piece in split_text(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    doc_id=path.stem,
                    title=title,
                    text=piece,
                )
            )
    return chunks