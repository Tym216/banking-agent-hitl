"""FAISS 向量库：构建、持久化、加载、检索。"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from banking_agent.rag.embeddings import Embedder
from banking_agent.rag.loader import Chunk


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("知识库为空，无法建索引")
        vecs = self._embedder.embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])  # 归一化向量 + 内积 = 余弦相似度
        index.add(vecs)
        self._index = index
        self._chunks = chunks

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if self._index is None:
            raise RuntimeError("索引未构建或未加载")
        qvec = self._embedder.embed([query])
        scores, ids = self._index.search(qvec, min(top_k, len(self._chunks)))
        return [
            (self._chunks[i], float(s))
            for i, s in zip(ids[0], scores[0])
            if i >= 0
        ]

    def save(self, index_dir: Path) -> None:
        assert self._index is not None
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / "index.faiss"))
        meta = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text} for c in self._chunks]
        (index_dir / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, index_dir: Path) -> bool:
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "chunks.json"
        if not (index_path.exists() and meta_path.exists()):
            return False
        self._index = faiss.read_index(str(index_path))
        self._chunks = [
            Chunk(**m) for m in json.loads(meta_path.read_text(encoding="utf-8"))
        ]
        return True
