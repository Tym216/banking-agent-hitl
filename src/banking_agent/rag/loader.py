"""知识库文档加载与切分:支持 markdown/json/txt 与 PDF(表格感知)。

流程: 读取 → 切分前剥离 frontmatter(进 metadata) → 切块 → 归一化。
Chunk.text 为归一化后的检索文本; display_text 保留原始内容供展示/审计。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 段落 → 换行 → 完整句子（绝不中途截断）→ 字符硬切（仅当单句超长时）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ""]

# JSON 纯元信息字段: 不参与检索正文, 也不参与元数据过滤, 但完整保留给 LLM 展示
_DEFAULT_METADATA_BLACKLIST = ["id", "source_url", "document_version",
                               "original_ids", "language"]

_FRONTMATTER_END_RE = re.compile(r"^---\n.*?\n---\n?", re.S)
_MARKER_RE = re.compile(r"\s*<!--\s*chunk\s*-->\s*")
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+.*$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str                       # 归一化后的检索文本
    display_text: str | None = None  # 展示/审计用原始内容; None 时回退到 text
    source_type: str = "paragraph"  # "paragraph" | "table" | "json"
    char_count: int = 0
    page_range: str | None = None   # e.g. "1-3"
    access_level: str = "all"       # "all" | "customer" | "staff" — 文件名前缀: cust_ / staff_ / 无前缀=all
    metadata: dict = field(default_factory=dict)  # frontmatter/JSON 顶层键; 用于过滤与溯源


def normalize_text(text: str) -> str:
    """全角→半角、统一常见标点、连续空白压成单个空格。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:                      # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:          # 全角 ASCII 区
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    s = "".join(out)
    s = s.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    s = s.replace("！", "!").replace("？", "?").replace("（", "(").replace("）", ")")
    return _WHITESPACE_RE.sub(" ", s).strip()


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析头部 YAML frontmatter（--- 包裹）。返回 (metadata, 剩余正文)；异常时原样返回。"""
    m = _FRONTMATTER_END_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(0).strip("-\n").strip()) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[m.end():]


def _json_search_text(obj: object, blacklist: set[str]) -> str:
    """拼接对象中所有非黑名单字段的值（字符串/数组/嵌套叶子），供检索。"""
    parts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if key in blacklist:
                    continue
                if isinstance(val, (dict, list)):
                    walk(val)
                else:
                    parts.append(str(val))
        elif isinstance(node, list):
            for item in node:
                walk(item)
        else:
            parts.append(str(node))

    walk(obj)
    return "\n".join(parts)


def _json_elements(path: Path, blacklist: set[str]) -> list[dict]:
    """JSON: 顶层数组每个对象=1 chunk; 正文=非黑名单字段值; 展示=完整对象; metadata=全部字段。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    objs = data if isinstance(data, list) else [data]
    elements: list[dict] = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        text = _json_search_text(obj, blacklist)
        if not text.strip():
            continue
        elements.append({
            "type": "json",
            "text": text,
            "display_text": json.dumps(obj, ensure_ascii=False, indent=2),
            "metadata": dict(obj),
            "pages": None,
        })
    return elements


def _table_to_text(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [str(cell) if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _read_pdf_pypdf(path: Path) -> list[dict]:
    """pypdf 回退解析器:只提纯文本,无表格。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return [{"type": "paragraph", "text": text, "pages": None}]


def _read_pdf(path: Path) -> list[dict]:
    """首选 pymupdf,其次 pdfplumber,回退 pypdf。任一层失败都向下回退。"""
    for parser in (_read_pdf_pymupdf, _read_pdf_pdfplumber):
        try:
            return parser(path)
        except (ImportError, Exception):
            continue
    try:
        return _read_pdf_pypdf(path)
    except Exception:
        return [{"type": "paragraph", "text": f"[无法解析: {path.name}]", "pages": None}]


def _read_pdf_pymupdf(path: Path) -> list[dict]:
    import pymupdf
    elements: list[dict] = []
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        page_num = str(i + 1)
        text = page.get_text("text")
        if text and text.strip():
            elements.append({"type": "paragraph", "text": text.strip(), "pages": page_num})
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for table in tabs.tables:
                    rows = table.extract()
                    if rows:
                        elements.append({"type": "table", "text": _table_to_text(rows), "pages": page_num})
        except Exception:
            pass
    if not elements:
        elements = [{"type": "paragraph", "text": "", "pages": None}]
    return elements


def _read_pdf_pdfplumber(path: Path) -> list[dict]:
    import pdfplumber
    elements: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = str(i + 1)
            text = page.extract_text(x_tolerance=1, y_tolerance=1)
            if text and text.strip():
                elements.append({"type": "paragraph", "text": text.strip(), "pages": page_num})
            tables = page.extract_tables()
            for table in tables:
                if not table or all(not any(cell for cell in row) for row in table):
                    continue
                elements.append({"type": "table", "text": _table_to_text(table), "pages": page_num})
    if not elements:
        elements = [{"type": "paragraph", "text": "", "pages": None}]
    return elements


def _split_heading_sections(text: str) -> list[str]:
    """按标题层级切块：每个标题及其后内容为一块（markdown 专用）。"""
    matches = list(_HEADING_RE.finditer(text))
    if len(matches) <= 1:
        return [text]
    parts: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append(text[m.start():end])
    return parts


def _recursive_split(text: str, chunk_size: int, overlap: int, sep_idx: int = 0) -> list[str]:
    """递归按分隔符优先级切分：段落 → 换行 → 完整句子，仅在单句超长时按字符硬切。"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if sep_idx >= len(_SEPARATORS) or not _SEPARATORS[sep_idx]:
        # 最后手段:按字符窗口切, 带 overlap（末位空分隔符=单字符粒度，等价硬切）
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    sep = _SEPARATORS[sep_idx]
    parts = text.split(sep)
    if len(parts) <= 1:
        return _recursive_split(text, chunk_size, overlap, sep_idx + 1)

    chunks: list[str] = []
    buf = ""
    for part in parts:
        if buf:
            candidate = buf + sep + part
        else:
            candidate = part

        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            sub = _recursive_split(part, chunk_size, overlap, sep_idx + 1)
            if sub:
                chunks.extend(sub[:-1])
                buf = sub[-1]
            else:
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _split_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """markdown 切分：<!-- chunk --> 标记优先；否则标题→段落→整句递归。

    标记模式下，首个标记前的头部内容（如标题）并入第一块，避免孤立标题污染检索。
    """
    marker_parts = [p.strip() for p in _MARKER_RE.split(text) if p.strip()]
    if len(marker_parts) > 1:
        if marker_parts[0].startswith(("#", "**", ">")) or "\n" not in marker_parts[0]:
            parts = [f"{marker_parts[0]}\n\n{marker_parts[1]}"] + marker_parts[2:]
        else:
            parts = marker_parts
    else:
        parts = [p.strip() for p in _split_heading_sections(text) if p.strip()]

    out: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            out.append(part)
        else:
            out.extend(_recursive_split(part, chunk_size, overlap))
    return out


def _split_element(text: str, chunk_size: int, overlap: int, source_type: str) -> list[str]:
    """按类型切分:表格/json 对象保持完整不切分;markdown 走标记/标题切分;其余递归分隔符切分。"""
    if source_type in ("table", "json"):
        return [text] if text.strip() else []
    if not text.strip():
        return []
    if source_type == "markdown":
        # 标记切分与文本长度无关（短文档也可能含 <!-- chunk -->）
        return _split_markdown(text, chunk_size, overlap)
    if len(text) <= chunk_size:
        return [text]
    return _recursive_split(text, chunk_size, overlap)


def _access_level(filename: str) -> str:
    if filename.startswith("cust_"):
        return "customer"
    if filename.startswith("staff_"):
        return "staff"
    return "all"


def load_knowledge_base(
    kb_dir: Path,
    chunk_size: int = 600,
    overlap: int = 50,
    *,
    normalize: bool = True,
    frontmatter: bool = True,
    metadata_blacklist: list[str] | None = None,
) -> list[Chunk]:
    """加载目录下全部受支持文档并切分。元数据黑名单字段只进展示/metadata，不进检索正文。"""
    blacklist = set(metadata_blacklist if metadata_blacklist is not None
                    else _DEFAULT_METADATA_BLACKLIST)
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*")):
        suffix = path.suffix.lower()
        elements: list[dict] = []
        if suffix == ".json":
            elements = _json_elements(path, blacklist)
        elif suffix in {".md", ".txt"}:
            raw = path.read_text(encoding="utf-8")
            meta: dict = {}
            if frontmatter and suffix == ".md":
                meta, raw = _parse_frontmatter(raw)
            source_type = "markdown" if suffix == ".md" else "paragraph"
            elements = [{"type": source_type, "text": raw, "pages": None,
                         "metadata": meta}]
        elif suffix == ".pdf":
            elements = _read_pdf(path)
        else:
            continue

        level = _access_level(path.name)
        chunk_id = 0
        for elem in elements:
            pieces = _split_element(elem["text"], chunk_size, overlap, elem["type"])
            for piece in pieces:
                chunks.append(Chunk(
                    doc_id=path.name,
                    chunk_id=chunk_id,
                    text=normalize_text(piece) if normalize else piece,
                    display_text=elem.get("display_text") or piece,
                    source_type=elem["type"] if elem["type"] != "markdown" else "paragraph",
                    char_count=len(piece),
                    page_range=elem.get("pages") or None,
                    access_level=level,
                    metadata=dict(elem.get("metadata") or {}),
                ))
                chunk_id += 1
    return chunks
