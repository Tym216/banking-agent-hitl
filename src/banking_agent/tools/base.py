"""工具声明式注册：每个工具带风险等级、最低角色、参数 schema 与授权钩子。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from banking_agent.auth.permissions import Role, User


class RiskLevel(str, Enum):
    NORMAL = "normal"      # 无需人工审批（写操作可配合 requires_confirmation 走客户确认）
    SENSITIVE = "sensitive"  # 必须人工审批


@dataclass
class ToolSpec:
    name: str
    description: str
    params_model: type[BaseModel]
    risk_level: RiskLevel
    required_role: Role
    handler: Callable[[User, BaseModel], dict[str, Any]]
    # 返回 None 表示通过，返回字符串表示拒绝原因（如越权查询他人账户）
    authorize: Callable[[User, dict[str, Any]], str | None] | None = None
    # 执行前客户确认（草稿→确认/取消/修改），与 staff 审批是两个正交维度
    requires_confirmation: bool = False

    def validate_args(self, args: dict[str, Any]) -> BaseModel:
        return self.params_model(**args)

    def execute(self, user: User, args: dict[str, Any]) -> dict[str, Any]:
        return self.handler(user, self.validate_args(args))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        return self._tools[name]

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())


__all__ = ["RiskLevel", "ToolSpec", "ToolRegistry", "ValidationError"]
