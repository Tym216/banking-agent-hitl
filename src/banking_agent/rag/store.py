"""FAISS 向量库 + SQLite metadata(含 FTS5 BM25)。

与业务库(agent.db)完全解耦：索引与 metadata 全部落在 index_dir 下。
检索管线：粗筛(每检索器独立) → 融合 → 元数据过滤 → 精排，见 retriever.py。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import faiss

from banking_agent.rag.bm25 import Bm25Index
from banking_agent.rag.embeddings import Embedder
from banking_agent.rag.loader import Chunk


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []
        self._meta_conn: sqlite3.Connection | None = None
        self._bm25: Bm25Index | None = None

    # ── build / metadata ────────────────────────────

    def build(self, chunks: list[Chunk], meta_db_path: Path) -> None:
        if not chunks:
            raise ValueError("知识库为空, 无法建索引")
        vecs = self._embedder.embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._index = index
        self._chunks = chunks

        meta_db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._meta_conn:
            self._meta_conn.close()
        conn = sqlite3.connect(str(meta_db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_meta (
                rowid     INTEGER PRIMARY KEY,
                doc_id    TEXT,
                chunk_id  INTEGER,
                access_level TEXT DEFAULT 'all',
                source_type  TEXT DEFAULT 'paragraph',
                char_count   INTEGER DEFAULT 0,
                page_range   TEXT,
                metadata_json TEXT DEFAULT '{}'   -- frontmatter/JSON 顶层键, 用于过滤与溯源
            )
        """)
        # 老库缺列回填（CREATE IF NOT EXISTS 不会为新列改老表）
        existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunk_meta)")}
        if "metadata_json" not in existing_cols:
            conn.execute("ALTER TABLE chunk_meta ADD COLUMN metadata_json TEXT DEFAULT '{}'")
        conn.execute("DELETE FROM chunk_meta")
        conn.executemany(
            "INSERT INTO chunk_meta (rowid, doc_id, chunk_id, access_level,"
            " source_type, char_count, page_range, metadata_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(i, c.doc_id, c.chunk_id, c.access_level, c.source_type,
              c.char_count, c.page_range, json.dumps(c.metadata, ensure_ascii=False))
             for i, c in enumerate(chunks)],
        )
        conn.commit()
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_access ON chunk_meta(access_level)")
        conn.commit()
        self._meta_conn = conn

        # BM25 索引与 metadata 同库（FTS5 虚表每次全量重建）
        self._bm25 = Bm25Index(conn)
        self._bm25.build([c.text for c in chunks])

    # ── 粗筛检索 ────────────────────────────────────

    def search(self, query: str, top_k: int = 50, min_score: float | None = None) -> list[tuple[Chunk, float]]:
        """FAISS 粗筛，可选最低分过滤（仅 embedding 域内使用）。"""
        if self._index is None:
            raise RuntimeError("索引未构建或未加载")
        qvec = self._embedder.embed([query])
        fetch_k = min(top_k * 2, len(self._chunks))  # 留余量抵消 min_score 过滤
        scores, ids = self._index.search(qvec, fetch_k)
        results: list[tuple[Chunk, float]] = []
        for i, s in zip(ids[0], scores[0]):
            if i < 0:
                continue
            if min_score is not None and float(s) < min_score:
                continue
            results.append((self._chunks[i], float(s)))
        return results[:top_k]

    def search_bm25(self, query: str, top_k: int = 50) -> list[tuple[Chunk, float]]:
        """FTS5 BM25 粗筛。分数为负值（越负越相关），由融合层归一化。"""
        if self._bm25 is None:
            raise RuntimeError("索引未构建或未加载")
        return [(self._chunks[rowid], score)
                for rowid, score in self._bm25.search(query, top_k)]

    # ── 元数据过滤（精确匹配，黑名单键不参与） ─────────

    def filter_chunks(
        self,
        candidates: list[tuple[Chunk, float]],
        filters: dict | None,
        blacklist: set[str],
    ) -> list[tuple[Chunk, float]]:
        """按 filters 精确匹配过滤候选。filters 为空或黑名单全命中时原样返回。"""
        if not filters:
            return candidates
        kept: list[tuple[Chunk, float]] = []
        for c, s in candidates:
            if all(self._match(c, key, val) for key, val in filters.items()
                   if key not in blacklist):
                kept.append((c, s))
        return kept

    @staticmethod
    def _match(chunk: Chunk, key: str, val: object) -> bool:
        if key == "access_level":
            return chunk.access_level == val
        if key == "doc_id":
            return chunk.doc_id == val
        # deprecated: valid_until_after 等旧特殊键不再支持，统一走 metadata 精确匹配
        return chunk.metadata.get(key) == val

    # ── save / load ─────────────────────────────────

    def save(self, index_dir: Path) -> None:
        assert self._index is not None
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / "index.faiss"))
        meta = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text,
                  "display_text": c.display_text, "source_type": c.source_type,
                  "char_count": c.char_count, "page_range": c.page_range,
                  "access_level": c.access_level, "metadata": c.metadata}
                 for c in self._chunks]
        (index_dir / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, index_dir: Path) -> bool:
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "chunks.json"
        if not (index_path.exists() and meta_path.exists()):
            return False
        self._index = faiss.read_index(str(index_path))
        chunks = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**m) for m in chunks]
        meta_db = index_dir / "chunks_meta.db"
        if not meta_db.exists():
            return False  # 索引不完整（缺 metadata/BM25），触发重建
        self._meta_conn = sqlite3.connect(str(meta_db))
        self._meta_conn.row_factory = sqlite3.Row
        self._meta_conn.execute("PRAGMA journal_mode=WAL")
        self._bm25 = Bm25Index(self._meta_conn)
        # 旧索引无 FTS 表 → 视为不完整，触发重建
        has_fts = self._meta_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
        ).fetchone()
        if not has_fts:
            return False
        return True
