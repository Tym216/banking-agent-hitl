"""精排器抽象：mock（测试/离线）与 cross_encoder（sentence-transformers）。

精排分数只用于重排候选，不参与 answer/clarify/refuse 判定（三态已废除）。
"""

from __future__ import annotations

from typing import Protocol

from banking_agent.config import RerankConfig
from banking_agent.rag.loader import Chunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        """返回与 chunks 等长的相关度分数（越大越相关）。"""
        ...


class MockReranker:
    """字符重叠率打分：确定性、离线，仅用于测试。"""

    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        q_set = set(query)
        scores = []
        for c in chunks:
            text = c.display_text or c.text
            if not text:
                scores.append(0.0)
                continue
            overlap = sum(1 for ch in q_set if ch in text)
            scores.append(overlap / max(len(q_set), 1))
        return scores


_MAX_CHARS = 2000  # bge-reranker 最大输入 ~512 tokens，取 2000 字符足够


class CrossEncoderReranker:
    """transformers 直载本地 rerank 模型目录（bge-reranker 系列，回归打分+sigmoid）。

    不用 sentence-transformers CrossEncoder：其 5.x 对纯文本模型强制走
    AutoProcessor，缺少 preprocessor_config.json 会加载失败。
    """

    def __init__(self, model_path: str) -> None:
        from transformers import (  # 延迟导入，重依赖
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()

    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [[query, (c.display_text or c.text)[:_MAX_CHARS]] for c in chunks]
        enc = self._tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        import torch

        with torch.no_grad():
            logits = self._model(**enc).logits
        scores = torch.sigmoid(logits).squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        return [float(s) for s in scores]


def create_reranker(cfg: RerankConfig) -> Reranker:
    if cfg.provider == "mock":
        return MockReranker()
    if cfg.provider == "cross_encoder":
        return CrossEncoderReranker(cfg.model)
    raise ValueError(f"未知 rerank provider: {cfg.provider}")
