"""LLM 客户端抽象：所有实现只需提供 chat()。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


@runtime_checkable
class LLMClient(Protocol):
    def chat(self, messages: list[Message], **kwargs: object) -> str:
        """输入 OpenAI 格式消息列表，返回助手回复文本。"""
        ...
