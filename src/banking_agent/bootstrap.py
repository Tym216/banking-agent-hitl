from __future__ import annotations

from pathlib import Path

from banking_agent.auth.permissions import Role, User
from banking_agent.config import AppConfig, load_config
from banking_agent.graph.workflow import AgentService
from banking_agent.llm import create_llm
from banking_agent.rag import Retriever, VectorStore, create_embedder, load_knowledge_base
from banking_agent.storage import AuditLogger, connect
from banking_agent.tools import ToolRegistry, create_default_registry

# 演示用户（真实系统中来自用户库/SSO）
DEMO_USERS: dict[str, User] = {
    "u_alice": User("u_alice", "Alice（客户）", Role.CUSTOMER, "ACC-001"),
    "u_bob": User("u_bob", "Bob（客户）", Role.CUSTOMER, "ACC-002"),
    "u_staff": User("u_staff", "柜员小王", Role.STAFF, can_approve=True),
}


def build_retriever(config: AppConfig, rebuild_index: bool = False) -> Retriever:
    embedder = create_embedder(config.embedding)
    store = VectorStore(embedder)
    index_dir = config.resolve_path(config.rag.index_dir)
    if rebuild_index or not store.load(index_dir):
        kb_dir = config.resolve_path(config.rag.kb_dir)
        chunks = load_knowledge_base(
            Path(kb_dir),
            config.rag.chunk_size,
            config.rag.chunk_overlap,
            normalize=config.rag.normalize_text,
            frontmatter=config.rag.frontmatter,
            metadata_blacklist=config.rag.metadata_blacklist,
        )
        store.build(chunks, index_dir / "chunks_meta.db")
        store.save(index_dir)
    return Retriever(store, config.rag)


def _setup_observability(config: AppConfig) -> None:
    """按配置设置 LangSmith 项目名；环境变量优先（便于临时覆盖）。"""
    import os

    project = config.observability.langsmith_project
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)  # 兼容旧版 SDK


def build_tool_registry(config: AppConfig) -> ToolRegistry:
    """按配置选择工具来源:local 进程内直调;mcp 走 MCP 协议消费。"""
    if config.tools.provider == "mcp":
        from banking_agent.mcp import MCPToolClient, create_mcp_tool_registry

        return create_mcp_tool_registry(MCPToolClient(config.mcp))
    return create_default_registry()


def create_service(
    config_path: str | Path | None = None,
    config: AppConfig | None = None,
    rebuild_index: bool = False,
) -> AgentService:
    cfg = config or load_config(config_path)
    _setup_observability(cfg)
    llm = create_llm(cfg.llm)
    retriever = build_retriever(cfg, rebuild_index)
    registry = build_tool_registry(cfg)
    audit = AuditLogger(connect(cfg.resolve_path(cfg.storage.db_path)))
    return AgentService(cfg, llm, retriever, registry, audit)
