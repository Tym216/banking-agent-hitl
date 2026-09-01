"""可视化单个文档的解析与切分结果, 用于填写验证集的 golden_snippets。

用法:
    python scripts/doc_chunks.py sample_fee-schedule.pdf
    → 输出该文档所有 chunk, 每段带 #编号 + 来源页码, 可直接复制到 YAML snippet

    python scripts/doc_chunks.py sample_fee-schedule.pdf --raw
    → 只输出纯文本, 不带头部元数据, 适合管道复制:
      python scripts/doc_chunks.py sample.pdf --raw | pbcopy

    python scripts/doc_chunks.py sample_fee-schedule.pdf --all
    → 输出全文(无切分), 适合看到文档全貌
"""

from __future__ import annotations

import argparse
from pathlib import Path

from banking_agent.rag.loader import load_knowledge_base


def show_chunks(kb_dir: Path, target: str, chunk_size: int, overlap: int, raw: bool = False) -> None:
    chunks = load_knowledge_base(kb_dir, chunk_size, overlap)
    doc_chunks = [c for c in chunks if c.doc_id == target]

    if not doc_chunks:
        print(f"未找到文档: {target}")
        print(f"知识库中的文档: {sorted({c.doc_id for c in chunks})}")
        return

    if raw:
        # 纯文本模式:只输出 chunk 内容,适合复制到 golden_snippets
        for c in doc_chunks:
            print(c.display_text or c.text)
            print("---")
        return

    print(f"## {target} ({len(doc_chunks)} chunks, {sum(c.char_count for c in doc_chunks)} chars)\n")
    for c in doc_chunks:
        pages = f"p{c.page_range}" if c.page_range else ""
        meta = f" meta={c.metadata}" if c.metadata else ""
        print(f"### [{c.chunk_id}] {pages} {c.source_type} ({c.char_count}c){meta}")
        print(c.display_text or c.text)
        print()


def show_full(kb_dir: Path, target: str, chunk_size: int) -> None:
    """显示文档全文(仅段落拼接), 含页码标记。"""
    from banking_agent.rag.loader import _read_pdf

    path = kb_dir / target
    if not path.exists():
        print(f"文件不存在: {path}")
        return

    elements = _read_pdf(path)
    for elem in elements:
        pages = f"[p{elem['pages']}]" if elem["pages"] else ""
        print(f"--- {elem['type']} {pages} ---")
        print(elem["text"])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc", help="文档文件名 (e.g. sample_fee-schedule.pdf)")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--all", action="store_true",
                        help="显示全文(不切分, 逐页带页码)")
    parser.add_argument("--raw", action="store_true",
                        help="只输出纯文本无元数据, 适合 | pbcopy")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(f"知识库目录不存在: {kb_dir}")
        exit(1)

    if args.all:
        show_full(kb_dir, args.doc, args.chunk_size)
    else:
        show_chunks(kb_dir, args.doc, args.chunk_size, args.overlap, raw=args.raw)
