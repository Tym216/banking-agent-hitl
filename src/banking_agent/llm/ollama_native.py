"""Ollama 原生 API 客户端。

与 openai_compat 的区别：原生 /api/chat 支持 think 参数控制推理模型的思考开关，
OpenAI 兼容端点会忽略该参数。
"""

from __future__ import annotations

import httpx

from banking_agent.config import LLMConfig
from banking_agent.llm.base import Message


class OllamaNativeClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._base = cfg.base_url.removesuffix("/v1").rstrip("/")

    def chat(self, messages: list[Message], **kwargs: object) -> str:
        payload: dict = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._cfg.temperature},
            **kwargs,
        }
        if self._cfg.think is not None:
            payload["think"] = self._cfg.think
        resp = httpx.post(
            f"{self._base}/api/chat", json=payload, timeout=self._cfg.timeout_s
        )
        resp.raise_for_status()
        return (resp.json()["message"]["content"] or "").strip()
