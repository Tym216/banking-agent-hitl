"""三态检索器：ANSWER（高置信）/ CLARIFY（低置信追问）/ REFUSE（未命中拒答）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from banking_agent.config import RAGConfig
from banking_agent.rag.loader import Chunk
from banking_agent.rag.store import VectorStore


class RetrievalDecision(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"


@dataclass
class RetrievalResult:
    decision: RetrievalDecision
    top_score: float
    hits: list[tuple[Chunk, float]] = field(default_factory=list)

    def context_text(self) -> str:
        return "\n\n".join(
            f"[{c.doc_id}#{c.chunk_id}] {c.text}" for c, _ in self.hits
        )

    def to_log_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "top_score": round(self.top_score, 4),
            "hits": [
                {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": round(s, 4)}
                for c, s in self.hits
            ],
        }


class Retriever:
    def __init__(self, store: VectorStore, cfg: RAGConfig) -> None:
        self._store = store
        self._cfg = cfg

    def retrieve(self, query: str) -> RetrievalResult:
        try:
            raw = self._store.search(query, self._cfg.top_k)
        except RuntimeError:
            # 索引不可用视为检索失败 → 拒答，而不是让 LLM 裸答
            return RetrievalResult(RetrievalDecision.REFUSE, top_score=0.0)
        if not raw:
            return RetrievalResult(RetrievalDecision.REFUSE, top_score=0.0)

        top_score = raw[0][1]
        if top_score >= self._cfg.high_threshold:
            hits = [(c, s) for c, s in raw if s >= self._cfg.low_threshold]
            return RetrievalResult(RetrievalDecision.ANSWER, top_score, hits)
        if top_score >= self._cfg.low_threshold:
            return RetrievalResult(RetrievalDecision.CLARIFY, top_score, raw[:1])
        return RetrievalResult(RetrievalDecision.REFUSE, top_score)
