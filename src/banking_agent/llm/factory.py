"""按配置构造 LLM 客户端。"""

from __future__ import annotations

from banking_agent.config import LLMConfig
from banking_agent.llm.base import LLMClient
from banking_agent.llm.mock import MockLLMClient
from banking_agent.llm.ollama_native import OllamaNativeClient
from banking_agent.llm.openai_compat import OpenAICompatClient


def create_llm(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "mock":
        return MockLLMClient()
    if cfg.provider == "openai_compat":
        return OpenAICompatClient(cfg)
    if cfg.provider == "ollama":
        return OllamaNativeClient(cfg)
    raise ValueError(f"未知 LLM provider: {cfg.provider}")
