"""构建 clean/ 索引目录:raw_text + clean_overrides 合并。

用法:
    venv-banking-agent/bin/python3 scripts/build_clean.py
    venv-banking-agent/bin/python3 scripts/build_clean.py --raw data/raw_text \
        --overrides data/clean_overrides --out data/clean

规则:
    - clean/ 每次全量重建(可放心删除)
    - clean_overrides/ 中同名文件优先,任何脚本都不修改该目录
    - raw_text/ 中未被覆盖的文件直接复制
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw_text")
    parser.add_argument("--overrides", default="data/clean_overrides")
    parser.add_argument("--out", default="data/clean")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    ov_dir = Path(args.overrides)
    out_dir = Path(args.out)

    if not raw_dir.is_dir():
        print(f"raw_text 目录不存在: {raw_dir} (先跑 parse_pdfs.py)")
        sys.exit(1)

    ov_dir.mkdir(parents=True, exist_ok=True)
    # 全量重建 clean/
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {f.name for f in ov_dir.iterdir() if f.is_file()}
    # 忽略扩展名匹配：hsbc_credit-card.json 可覆盖 hsbc_credit-card.txt
    override_stems = {Path(name).stem: name for name in overrides}
    used_override, copied, orphan = 0, 0, 0

    for src in sorted(raw_dir.iterdir()):
        if not src.is_file():
            continue
        ov_name = override_stems.get(src.stem)
        if ov_name:
            shutil.copy2(ov_dir / ov_name, out_dir / ov_name)
            print(f"  ✎ {src.name} ← {ov_name} (override 覆盖)")
            used_override += 1
            override_stems.pop(src.stem, None)
        else:
            shutil.copy2(src, out_dir / src.name)
            print(f"  ✓ {src.name} (raw 复制)")
            copied += 1

    # clean_overrides 里有多余的文件(raw_text 里没有对应 stem)
    for name in sorted(override_stems.values()):
        shutil.copy2(ov_dir / name, out_dir / name)
        print(f"  + {name} (override 独有,已加入)")
        orphan += 1

    print(f"\n完成: {used_override} 手工, {copied} 复制, {orphan} override 独有 → {out_dir}")


if __name__ == "__main__":
    main()
