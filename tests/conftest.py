"""测试装配：Mock LLM + Mock Embedding，全部离线、确定性运行。"""

from __future__ import annotations

import os

import pytest

# 单元测试全是 Mock 调用，tracing 只有噪音；无论全局环境怎么配都强制关闭
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from banking_agent.bootstrap import create_service
from banking_agent.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    RAGConfig,
    StorageConfig,
)
from banking_agent.tools import mock_bank


@pytest.fixture()
def service(tmp_path):
    mock_bank.reset()
    cfg = AppConfig(
        llm=LLMConfig(provider="mock"),
        embedding=EmbeddingConfig(provider="mock", dim=512),
        rag=RAGConfig(
            index_dir=str(tmp_path / "index"),
            # mock embedder 是字符 bigram 哈希袋，相似度整体偏低，阈值相应下调
            high_threshold=0.22,
            low_threshold=0.13,
        ),
        storage=StorageConfig(
            db_path=str(tmp_path / "audit.db"),
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        ),
    )
    return create_service(config=cfg, rebuild_index=True)
