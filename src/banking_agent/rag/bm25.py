"""BM25 检索：基于 SQLite FTS5（零新依赖，离线可用）。

中文分词：CJK 连续串按 2-gram 预切分后写入 FTS5，查询同样切分，
避免 unicode61 按单字切分导致精度差。分数为负值（越负越相关），
排序用 ORDER BY bm25 ASC；归一化交给融合层处理。
"""

from __future__ import annotations

import re
import sqlite3

_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")

_FTS_TABLE = "chunk_fts"


def tokenize(text: str) -> str:
    """把文本转成 FTS5 可索引的 token 串：ASCII 词原样，CJK 按 bigram。"""
    tokens: list[str] = []
    pos = 0
    for m in _ASCII_WORD.finditer(text):
        start, end = m.span()
        if start > pos:
            _append_cjk(text[pos:start], tokens)
        tokens.append(m.group())
        pos = end
    _append_cjk(text[pos:], tokens)
    return " ".join(tokens)


def _append_cjk(run: str, tokens: list[str]) -> None:
    for m in _CJK_RUN.finditer(run):
        seq = m.group()
        if len(seq) == 1:
            tokens.append(seq)
        else:
            tokens.extend(seq[i : i + 2] for i in range(len(seq) - 1))


class Bm25Index:
    """FTS5 包装。rowid 与 chunks 列表下标对齐，便于映射回 Chunk。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def build(self, texts: list[str]) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {_FTS_TABLE}")
        self._conn.execute(
            f"CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5(text)"
        )
        self._conn.executemany(
            f"INSERT INTO {_FTS_TABLE} (rowid, text) VALUES (?, ?)",
            [(i, tokenize(t)) for i, t in enumerate(texts)],
        )
        self._conn.commit()

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """返回 [(rowid, score)]，按相关度降序；分数为 FTS5 bm25（负值，越负越相关）。

        MATCH 用 OR 连接 token（AND 对多词查询召回为零），命中按 bm25 打分排序。
        """
        q = tokenize(query)
        if not q:
            return []
        match_expr = " OR ".join(q.split())
        try:
            rows = self._conn.execute(
                f"SELECT rowid, bm25({_FTS_TABLE}) AS score"
                f" FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ?"
                f" ORDER BY score ASC LIMIT ?",
                (match_expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # 查询词全是停用词等导致 MATCH 语法错误
        return [(int(r[0]), float(r[1])) for r in rows]
