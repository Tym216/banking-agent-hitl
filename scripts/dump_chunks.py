"""导出知识库切分结果, 用于可视化检查 chunk 来源与边界。

用法:
    python scripts/dump_chunks.py                           # 输出 JSONL
    python scripts/dump_chunks.py --format markdown          # 按文档分组, 便于浏览
    python scripts/dump_chunks.py --chunk-size 800 --overlap 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from banking_agent.rag.loader import load_knowledge_base


def _jsonl(kb_dir: Path, chunk_size: int, overlap: int) -> None:
    out = Path("data/chunks_dump.jsonl")
    chunks = load_knowledge_base(kb_dir, chunk_size, overlap)
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            preview = c.text[:200].replace("\n", "\\n")
            f.write(json.dumps({
                "doc_id": c.doc_id, "chunk_id": c.chunk_id,
                "char_count": c.char_count, "source_type": c.source_type,
                "page_range": c.page_range, "preview": preview,
            }, ensure_ascii=False) + "\n")
    print(f"写入 {len(chunks)} chunks → {out}")


def _markdown(kb_dir: Path, chunk_size: int, overlap: int) -> None:
    out = Path("data/chunks_dump.md")
    chunks = load_knowledge_base(kb_dir, chunk_size, overlap)
    by_doc: dict[str, list] = {}
    total_chars = 0
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
        total_chars += c.char_count

    lines = [
        f"# Chunk Dump ({len(chunks)} chunks, {total_chars} chars total)\n",
        f"chunk_size={chunk_size}, overlap={overlap}\n",
    ]
    for doc_id in sorted(by_doc):
        doc_chunks = by_doc[doc_id]
        doc_chars = sum(c.char_count for c in doc_chunks)
        lines.append(f"## {doc_id} ({len(doc_chunks)} chunks, {doc_chars} chars)\n")
        lines.append("| # | Type | Chars | Pages | Preview |")
        lines.append("|---|---|---|---|---|")
        for c in doc_chunks:
            preview = c.text[:100].replace("\n", " ").replace("|", "\\|")
            pages = c.page_range or "-"
            lines.append(f"| {c.chunk_id} | {c.source_type} | {c.char_count} | {pages} | {preview} |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {len(chunks)} chunks, {len(by_doc)} documents → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--format", choices=["jsonl", "markdown"], default="jsonl")
    args = parser.parse_args()
    kb_dir = Path(args.kb_dir)
    if not kb_dir.exists():
        print(f"知识库目录不存在: {kb_dir}")
        exit(1)
    {"jsonl": _jsonl, "markdown": _markdown}[args.format](kb_dir, args.chunk_size, args.overlap)
