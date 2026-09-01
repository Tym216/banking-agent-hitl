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
            kb_dir="tests/data/mock_kb",  # 隔离真实文档,只测 mock 知识库
            index_dir=str(tmp_path / "index"),
            chunk_size=400,  # mock 文档极短,保持小块以维持测试向量分布
            # mock embedder 是字符 bigram 哈希袋，400 字 chunk 下：
            # 真实命中 ~0.27-0.39，无关噪声 ~0.06-0.12；min_score=0.2 过滤噪声
            # （对应线上 embedding 0.15 的粗滤语义，值随文档/embedder 标定）
            retrievers={"type": "embedding", "top_k": 20, "min_score": 0.2},
        ),
        storage=StorageConfig(
            db_path=str(tmp_path / "audit.db"),
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        ),
    )
    return create_service(config=cfg, rebuild_index=True)
