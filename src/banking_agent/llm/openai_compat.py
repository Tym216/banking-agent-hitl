"""OpenAI-compatible 客户端：一个实现覆盖 OpenAI / vLLM / Ollama。

vLLM 与 Ollama 均原生暴露 /v1/chat/completions，切换来源只需改配置中的
base_url 与 model，无需改代码。
"""

from __future__ import annotations

import re

from openai import OpenAI

from banking_agent.config import LLMConfig
from banking_agent.llm.base import Message

# Qwen3 等推理模型会输出 <think>...</think> 思考块，下游解析前统一剥离
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,  # vLLM/Ollama 不校验 key，占位值即可
            timeout=cfg.timeout_s,
        )

    def chat(self, messages: list[Message], **kwargs: object) -> str:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._cfg.temperature,
            **kwargs,  # type: ignore[arg-type]
        )
        content = resp.choices[0].message.content or ""
        return _THINK_RE.sub("", content).strip()
