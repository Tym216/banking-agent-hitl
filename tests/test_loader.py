"""测试文档加载与切分：frontmatter / JSON 三轨 / marker / 句子边界 / 归一化。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from banking_agent.rag.loader import (
    Chunk,
    _parse_frontmatter,
    _split_element,
    load_knowledge_base,
    normalize_text,
)


_TMP_HOLD: list = []  # 持有 TemporaryDirectory 引用，防止提前 GC


def _tmp_kb(files: dict[str, str]) -> Path:
    tmp = tempfile.TemporaryDirectory()
    _TMP_HOLD.append(tmp)
    for name, content in files.items():
        (Path(tmp.name) / name).write_text(content, encoding="utf-8")
    return Path(tmp.name)


def test_loads_markdown():
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "test.md"
        md.write_text("# 标题\n\n第一段。\n\n第二段。", encoding="utf-8")
        chunks = load_knowledge_base(Path(tmp), chunk_size=600, overlap=50)
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "test.md"
        assert chunks[0].char_count > 0
        assert chunks[0].source_type == "paragraph"


def test_chunk_metadata_fields():
    chunk = Chunk(doc_id="a.pdf", chunk_id=0, text="hello world",
                  source_type="table", char_count=11, page_range="3",
                  metadata={"k": "v"})
    assert chunk.source_type == "table"
    assert chunk.char_count == 11
    assert chunk.page_range == "3"
    assert chunk.metadata == {"k": "v"}


def test_chunk_defaults():
    """新字段都有默认值,兼容旧 chunks.json 反序列化。"""
    chunk = Chunk(doc_id="x", chunk_id=0, text="t")
    assert chunk.source_type == "paragraph"
    assert chunk.display_text is None
    assert chunk.metadata == {}


def test_skips_unsupported_files():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "img.png").write_text("nope", encoding="utf-8")
        (Path(tmp) / "ok.md").write_text("hello", encoding="utf-8")
        chunks = load_knowledge_base(Path(tmp))
        assert len(chunks) == 1
        assert chunks[0].doc_id == "ok.md"


def test_table_element_not_split():
    """表格块不被切分,即使超过 chunk_size。"""
    long_table = "\n".join([" | ".join(["cell"] * 30) for _ in range(10)])
    assert len(long_table) > 600
    pieces = _split_element(long_table, chunk_size=400, overlap=50, source_type="table")
    assert len(pieces) == 1
    assert pieces[0] == long_table


def test_paragraph_splitting():
    short = "只有一段。" * 20
    pieces = _split_element(short, chunk_size=600, overlap=50, source_type="paragraph")
    assert len(pieces) == 1

    long = "长文本。" * 200
    pieces = _split_element(long, chunk_size=600, overlap=50, source_type="paragraph")
    assert len(pieces) >= 2


def test_normalize_text():
    assert normalize_text("ＡＢＣ，中文　标点  连续空格") == "ABC,中文 标点 连续空格"
    assert normalize_text("　") == ""
    assert normalize_text("  多   个    空格  ") == "多 个 空格"


def test_frontmatter_parsed_to_metadata():
    md = """---
card_types:
  - HSBC Red
source_url: https://example.com
document_version: "2025-06"
---

# 正文标题

正文内容。"""
    kb = _tmp_kb({"a.md": md})
    chunks = load_knowledge_base(kb)
    assert len(chunks) >= 1
    c = chunks[0]
    # frontmatter 只进 metadata
    assert c.metadata["card_types"] == ["HSBC Red"]
    assert c.metadata["source_url"] == "https://example.com"
    # 不进检索正文，也不进展示正文
    assert "source_url" not in c.text
    assert "HSBC Red" not in (c.display_text or c.text)


def test_frontmatter_no_frontmatter_ok():
    meta, body = _parse_frontmatter("没有 frontmatter 的正文。")
    assert meta == {}
    assert "正文" in body


def test_marker_splitting():
    """<!-- chunk --> 标记优先切分。"""
    md = """# 标题

<!-- chunk -->
第一段内容。

<!-- chunk -->
第二段内容。

<!-- chunk -->
第三段内容。"""
    kb = _tmp_kb({"a.md": md})
    chunks = load_knowledge_base(kb, chunk_size=600, overlap=50)
    texts = [(c.display_text or c.text) for c in chunks]
    assert len(texts) == 3
    assert any("第一段" in t for t in texts)
    assert any("第二段" in t for t in texts)


def test_json_chunking_three_track():
    """JSON 三轨：text 无黑名单字段 / display_text 完整 / metadata 全字段。"""
    data = [
        {"id": "x1", "title": "转账限额", "content": "每日转账限额 50 万港元。",
         "source_url": "https://x", "language": "zh-HK"},
        {"id": "x2", "title": "挂失步骤", "steps": ["第一步", "第二步"], "content": "请立即联系我们。"},
    ]
    kb = _tmp_kb({"b.json": json.dumps(data, ensure_ascii=False)})
    chunks = load_knowledge_base(kb)
    assert len(chunks) == 2
    c0, c1 = chunks[0], chunks[1]
    # text: 非黑名单字段拼接（不含 id/source_url/language）
    assert "转账限额" in c0.text and "50 万港元" in c0.text
    assert "x1" not in c0.text and "https://x" not in c0.text
    assert "zh-HK" not in c0.text
    # display_text: 完整 JSON 对象
    assert '"source_url"' in (c0.display_text or "")
    assert '"id": "x1"' in (c0.display_text or "")
    # metadata: 全部字段
    assert set(c0.metadata) == {"id", "title", "content", "source_url", "language"}
    assert c0.source_type == "json"
    # 嵌套数组也进检索正文
    assert "第一步" in c1.text


def test_blacklist_excluded_from_text():
    """自定义黑名单生效。"""
    data = [{"id": "x1", "secret_key": "s3cr3t", "content": "公开内容"}]
    kb = _tmp_kb({"b.json": json.dumps(data, ensure_ascii=False)})
    chunks = load_knowledge_base(kb, metadata_blacklist=["id", "secret_key"])
    assert "s3cr3t" not in chunks[0].text
    assert "公开内容" in chunks[0].text
    # 黑名单字段仍保留在 metadata
    assert chunks[0].metadata["secret_key"] == "s3cr3t"


def test_no_mid_sentence_cut():
    """切分不跨句截断：块边界永远在完整句子之间。"""
    import re

    sents = [f"第{i}句内容。" for i in range(1, 60)]
    text = "".join(sents)
    chunks = _split_element(text, chunk_size=100, overlap=10, source_type="paragraph")
    assert len(chunks) >= 2
    all_nums: list[int] = []
    for c in chunks:
        nums = [int(m) for m in re.findall(r"第(\d+)句", c)]
        assert nums, f"块内没有完整句子: {c[:30]}"
        # 块内句子编号连续升序（未被截断出半句）
        assert nums == list(range(nums[0], nums[-1] + 1))
        all_nums.extend(nums)
    assert set(all_nums) == set(range(1, 60))  # 全部句子无遗漏


def test_unpunctuated_long_line_hard_cut():
    """无任何分隔符的长文本（无标点连续串）不崩溃，按字符窗口硬切。"""
    text = "中文无标点连续文本" * 100  # 800 字符，无句号/空格/换行
    chunks = _split_element(text, chunk_size=200, overlap=20, source_type="paragraph")
    assert len(chunks) >= 4
    assert all(len(c) <= 200 for c in chunks)


def test_normalization_keeps_display_text():
    """归一化只影响 text，display_text 保留原始内容。"""
    md = "# 标题\n\n年费：４８０ 港元！"
    kb = _tmp_kb({"a.md": md})
    chunks = load_knowledge_base(kb, chunk_size=600, overlap=50, normalize=True)
    c = chunks[0]
    assert "480" in c.text and ":" in c.text
    assert "４８０" in (c.display_text or c.text)


def test_normalize_off():
    md = "# 标题\n\n年费：４８０ 港元！"
    kb = _tmp_kb({"a.md": md})
    chunks = load_knowledge_base(kb, chunk_size=600, overlap=50, normalize=False)
    assert "４８０" in chunks[0].text
