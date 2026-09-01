"""检索管线：多检索器粗筛 → 融合(RRF/加权和) → 元数据过滤 → 精排 → hits。

三态判定已废除：hits 为空即拒答；回答/追问由 LLM 自主判断（见 nodes.policy_qa）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from banking_agent.config import RAGConfig
from banking_agent.rag.loader import Chunk
from banking_agent.rag.rerank import create_reranker
from banking_agent.rag.store import VectorStore


@dataclass
class RetrievalResult:
    hits: list[tuple[Chunk, float]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.hits

    def context_text(self) -> str:
        return "\n\n".join(
            f"[{c.doc_id}#{c.chunk_id}] {c.display_text or c.text}" for c, _ in self.hits
        )

    def to_log_dict(self) -> dict:
        return {
            "decision": "refuse" if self.is_empty() else "answer",
            "top_score": round(self.hits[0][1], 4) if self.hits else 0.0,
            "hits": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": round(s, 4)}
                for c, s in self.hits
            ],
        }


def _s2t(query: str) -> str:
    try:
        from opencc import OpenCC
        return OpenCC("s2t").convert(query)
    except ImportError:
        return query


def _rrf_fuse(lists: list[list[tuple[Chunk, float]]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion：按排名投票，不依赖分数域。"""
    scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], Chunk] = {}
    for lst in lists:
        for rank, (chunk, _) in enumerate(lst):
            key = (chunk.doc_id, chunk.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunks[key] = chunk
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(chunks[key], score) for key, score in ordered]


def _minmax_norm(lst: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    """分数 min-max 归一化到 [0,1]（加权和融合前用，抹平 embedding/BM25 分数域差异）。"""
    if not lst:
        return []
    vals = [s for _, s in lst]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [(c, 1.0) for c, _ in lst]
    return [(c, (s - lo) / (hi - lo)) for c, s in lst]


class Retriever:
    def __init__(self, store: VectorStore, cfg: RAGConfig) -> None:
        self._store = store
        self._cfg = cfg
        self._reranker = create_reranker(cfg.rerank) if cfg.rerank.enabled else None

    def _coarse(self, query: str) -> list[list[tuple[Chunk, float]]]:
        """每个检索器独立粗筛（top_k 默认 50）。embedding 带 min_score，bm25 仅截断。"""
        ranked: list[list[tuple[Chunk, float]]] = []
        for item in self._cfg.retrievers_resolved:
            if item.type == "embedding":
                ranked.append(self._store.search(query, item.top_k, min_score=item.min_score))
            else:
                ranked.append(self._store.search_bm25(query, item.top_k))
        return ranked

    def _fuse(self, ranked: list[list[tuple[Chunk, float]]]) -> list[tuple[Chunk, float]]:
        if self._cfg.fusion.method == "rrf":
            return _rrf_fuse(ranked, self._cfg.fusion.rrf_k)
        # weighted_sum：各检索器分数归一化后按权重加权
        weights = [r.weight for r in self._cfg.retrievers_resolved]
        merged: dict[tuple[str, int], float] = {}
        chunks: dict[tuple[str, int], Chunk] = {}
        for lst, w in zip(ranked, weights):
            for c, s in _minmax_norm(lst):
                key = (c.doc_id, c.chunk_id)
                merged[key] = merged.get(key, 0.0) + w * s
                chunks[key] = c
        ordered = sorted(merged.items(), key=lambda kv: -kv[1])
        return [(chunks[key], score) for key, score in ordered]

    def retrieve(self, query: str, filters: dict | None = None) -> RetrievalResult:
        try:
            q = _s2t(query) if self._cfg.query_to_traditional else query
            ranked = self._coarse(q)
        except RuntimeError:
            return RetrievalResult()
        if not any(ranked):
            return RetrievalResult()

        fused = self._fuse(ranked)
        if not fused:
            return RetrievalResult()

        # 元数据过滤（默认关；过滤后为空则回退未过滤结果，避免误伤）
        if self._cfg.enable_metadata_filter:
            blacklist = set(self._cfg.metadata_blacklist)
            filtered = self._store.filter_chunks(fused, filters, blacklist)
            if not filtered:
                print(f"[retriever] 元数据过滤后候选为空，回退未过滤结果")
                filtered = fused
            fused = filtered

        # 精排：对全部候选重排，取 rerank.top_k 作为候选池，最终截断到 rag.top_k
        if self._reranker is not None:
            chunks = [c for c, _ in fused]
            scores = self._reranker.rerank(q, chunks)
            fused = sorted(zip(chunks, scores), key=lambda cs: -cs[1])[
                : self._cfg.rerank.top_k
            ]

        return RetrievalResult(hits=fused[: self._cfg.top_k])
