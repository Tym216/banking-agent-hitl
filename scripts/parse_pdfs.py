"""解析 knowledge_base/ 下所有 PDF → raw_text/*.txt。

用法:
    venv-banking-agent/bin/python3 scripts/parse_pdfs.py
    venv-banking-agent/bin/python3 scripts/parse_pdfs.py --kb data/knowledge_base --out data/raw_text
    venv-banking-agent/bin/python3 scripts/parse_pdfs.py --force   # 覆盖已有输出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banking_agent.rag.loader import _read_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="data/knowledge_base")
    parser.add_argument("--out", default="data/raw_text")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已有输出文件")
    args = parser.parse_args()

    kb_dir = Path(args.kb)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not kb_dir.is_dir():
        print(f"知识库目录不存在: {kb_dir}")
        sys.exit(1)

    converted, skipped, failed = 0, 0, 0
    for pdf in sorted(kb_dir.glob("*.pdf")):
        out_path = out_dir / f"{pdf.stem}.txt"
        if out_path.exists() and not args.force:
            print(f"  ⏭ {pdf.name} (已存在, --force 覆盖)")
            skipped += 1
            continue
        try:
            elements = _read_pdf(pdf)
            parts = []
            for elem in elements:
                if elem["type"] == "table":
                    parts.append(f"[TABLE]\n{elem['text']}")
                else:
                    parts.append(elem["text"])
            text = "\n\n".join(parts)
            out_path.write_text(text, encoding="utf-8")
            print(f"  ✓ {pdf.name} → {out_path.name} ({len(text)} chars)")
            converted += 1
        except Exception as e:
            print(f"  ✗ {pdf.name}: {e}")
            failed += 1

    print(f"\n完成: {converted} 转换, {skipped} 跳过, {failed} 失败 → {out_dir}")


if __name__ == "__main__":
    main()
