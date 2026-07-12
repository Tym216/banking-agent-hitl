"""Embedding 抽象与三种实现：mock / sentence-transformers / OpenAI-compatible API。

所有实现输出 L2 归一化向量，FAISS 用内积索引即等价余弦相似度。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from banking_agent.config import EmbeddingConfig


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 shape=(n, dim) 的归一化 float32 矩阵。"""
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class MockEmbedder:
    """字符 bigram 哈希袋向量：确定性、离线，相似文本得到相似向量。仅用于测试。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j in range(len(text) - 1):
                bigram = text[j : j + 2]
                h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
                mat[i, h % self.dim] += 1.0
        return _normalize(mat)


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，重依赖

        self._model = SentenceTransformer(model_name)
        get_dim = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatEmbedder:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        import os

        from openai import OpenAI

        self._client = OpenAI(
            base_url=cfg.base_url or None,
            api_key=os.environ.get(cfg.api_key_env, "EMPTY"),
        )
        self._model = cfg.model
        self.dim = cfg.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        mat = np.array([d.embedding for d in resp.data], dtype=np.float32)
        self.dim = mat.shape[1]
        return _normalize(mat)


def create_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "mock":
        return MockEmbedder(cfg.dim)
    if cfg.provider == "sentence_transformers":
        return SentenceTransformersEmbedder(cfg.model)
    if cfg.provider == "openai_compat":
        return OpenAICompatEmbedder(cfg)
    raise ValueError(f"未知 embedding provider: {cfg.provider}")
