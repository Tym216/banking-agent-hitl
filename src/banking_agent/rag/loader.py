"""知识库文档加载与切分：支持 markdown/txt 与 PDF。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str      # 来源文件名
    chunk_id: int
    text: str


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """先按段落聚合，超长再滑窗切分。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        while len(p) > chunk_size:
            chunks.append(p[:chunk_size])
            p = p[chunk_size - overlap :]
        buf = p
    if buf:
        chunks.append(buf)
    return chunks


def load_knowledge_base(kb_dir: Path, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        for i, piece in enumerate(_split(text, chunk_size, overlap)):
            chunks.append(Chunk(doc_id=path.name, chunk_id=i, text=piece))
    return chunks
