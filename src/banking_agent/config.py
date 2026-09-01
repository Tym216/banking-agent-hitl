"""配置加载：YAML 文件 + 环境变量覆盖（BANKING_AGENT_ 前缀）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    provider: Literal["mock", "openai_compat", "ollama"] = "mock"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.1
    timeout_s: int = 60
    # 仅 ollama：思考模型的推理开关；None 表示用模型默认值
    think: bool | None = None

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "EMPTY")


class EmbeddingConfig(BaseModel):
    provider: Literal["mock", "sentence_transformers", "openai_compat"] = "mock"
    model: str = "BAAI/bge-small-zh-v1.5"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    dim: int = 512


class RetrieverItemConfig(BaseModel):
    """单个检索器配置。min_score 仅 embedding 域内生效（BM25 分数为负，禁止设置）。"""

    type: Literal["embedding", "bm25"] = "embedding"
    weight: float = 1.0
    top_k: int = 50          # 粗筛候选数，为过滤/精排留余量
    min_score: float | None = None   # embedding 粗筛最低分；None=不按分数过滤

    @model_validator(mode="after")
    def _check(self) -> "RetrieverItemConfig":
        if self.weight <= 0:
            raise ValueError("weight 必须为正数")
        if self.top_k < 1:
            raise ValueError("top_k 至少为 1")
        if self.type == "bm25" and self.min_score is not None:
            raise ValueError("bm25 检索器不允许设置 min_score（FTS5 分数为负，无法共用 embedding 阈值）")
        if self.min_score is not None and not (0.0 <= self.min_score < 1.0):
            raise ValueError("min_score 必须在 [0, 1) 或为 None")
        return self


class FusionConfig(BaseModel):
    """多检索器融合方式：RRF（默认）或加权求和。"""

    method: Literal["rrf", "weighted_sum"] = "rrf"
    rrf_k: int = 60

    @model_validator(mode="after")
    def _check(self) -> "FusionConfig":
        if self.rrf_k < 1:
            raise ValueError("rrf_k 至少为 1")
        return self


class RerankConfig(BaseModel):
    """精排配置。分数只用于排序，不参与三态判定（三态已废除）。"""

    enabled: bool = False
    provider: Literal["mock", "cross_encoder"] = "cross_encoder"
    model: str = "data/models/bge-reranker-v2-m3"
    top_k: int = 5            # 精排后保留的候选池大小（须 >= rag.top_k）

    @model_validator(mode="after")
    def _check(self) -> "RerankConfig":
        if self.top_k < 1:
            raise ValueError("rerank.top_k 至少为 1")
        return self


class RAGConfig(BaseModel):
    kb_dir: str = "data/clean"
    index_dir: str = "data/index"
    chunk_size: int = 600
    chunk_overlap: int = 50
    top_k: int = 3            # 最终返回给 LLM 的条数
    # deprecated: 双阈值三态判定已废除（追问交给 LLM 自主判断），
    # 字段仅保留以兼容旧配置/测试传参，逻辑不再使用。
    high_threshold: float | None = None
    low_threshold: float | None = None
    query_to_traditional: bool = False  # 简体查询自动转繁体后再检索
    normalize_text: bool = True         # 切块后归一化（全角→半角等）
    frontmatter: bool = True            # md frontmatter 解析（只进 metadata）
    enable_metadata_filter: bool = False  # 元数据过滤开关（默认关，对比效果用）
    metadata_blacklist: list[str] = Field(
        default_factory=lambda: ["id", "source_url", "document_version",
                                 "original_ids", "language"]
    )
    # 单一检索器：dict；混合检索：list（每个元素一个检索器配置）
    retrievers: list[RetrieverItemConfig] | None = None
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)

    @field_validator("retrievers", mode="before")
    @classmethod
    def _normalize_retrievers(cls, v: Any) -> Any:
        """dict 视为单一检索器，统一归一为 list，避免 union 校验噪音。"""
        if isinstance(v, dict):
            return [v]
        return v

    @model_validator(mode="after")
    def _check_bounds(self) -> "RAGConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size，否则切分会死循环")
        if (self.high_threshold is not None and self.low_threshold is not None
                and self.low_threshold > self.high_threshold):
            raise ValueError("low_threshold 不能大于 high_threshold")
        if self.retrievers is not None and not self.retrievers:
            raise ValueError("retrievers 列表不能为空")
        if self.rerank.enabled and self.rerank.top_k < self.top_k:
            raise ValueError("rerank.top_k 必须 >= rag.top_k（精排候选池需覆盖最终返回数）")
        return self

    @property
    def retrievers_resolved(self) -> list[RetrieverItemConfig]:
        """返回检索器列表：None 用默认 embedding。"""
        if self.retrievers is None:
            return [RetrieverItemConfig(type="embedding")]
        return self.retrievers


class ToolsConfig(BaseModel):
    # local: 进程内直接调用(测试/离线默认)
    # mcp:   工具由 MCP server 提供,agent 侧走 MCP 协议消费
    provider: Literal["local", "mcp"] = "local"


class MCPConfig(BaseModel):
    # stdio: agent 自动拉起 server 子进程,零配置(默认)
    # http:  连接独立部署的 streamable HTTP server(先手动启动 --http)
    transport: Literal["stdio", "http"] = "stdio"
    url: str = "http://localhost:8000/mcp"
    timeout_s: int = 30
    # HTTP 模式 Bearer token 来源环境变量名(server 与 client 用同一个 token)
    auth_token_env: str = "MCP_AUTH_TOKEN"


class StorageConfig(BaseModel):
    db_path: str = "data/banking_agent.db"
    checkpoint_db_path: str = "data/checkpoints.db"


class ObservabilityConfig(BaseModel):
    # 非空时启动自动设置 LANGSMITH_PROJECT / LANGCHAIN_PROJECT。
    # API key 与 LANGSMITH_TRACING 属于机密/全局开关，仍由环境变量（如 ~/.bashrc）管理。
    langsmith_project: str = ""


class AppConfig(BaseSettings):
    """全局配置。环境变量覆盖示例：BANKING_AGENT_LLM__PROVIDER=openai_compat"""

    model_config = SettingsConfigDict(
        env_prefix="BANKING_AGENT_", env_nested_delimiter="__", extra="ignore"
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载 YAML 配置；环境变量优先级高于文件。path 为空时用默认 configs/config.yaml。"""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # BaseSettings 中 init kwargs(即上面的 YAML)优先级高于环境变量，
    # 这里手动把 BANKING_AGENT_ 前缀的 env 深合并进 YAML，实现"环境变量覆盖文件"
    return AppConfig(**_deep_merge(_env_overrides(), data))


def _env_overrides() -> dict[str, Any]:
    """把 BANKING_AGENT_A__B__C=value 解析为 {'a': {'b': {'c': 'value'}}}。"""
    prefix = "BANKING_AGENT_"
    out: dict[str, Any] = {}
    for key in sorted(os.environ):
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = os.environ[key]
    return out


def _deep_merge(overrides: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(value, merged[key])
        else:
            merged[key] = value
    return merged
