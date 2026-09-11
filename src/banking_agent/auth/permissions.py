"""权限模型：角色分级 + 工具级授权 + 强制审批裁决。

裁决完全由框架代码完成，不依赖 LLM 输出，因此 prompt 注入无法绕过。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from banking_agent.tools.base import ToolSpec


class Role(str, Enum):
    CUSTOMER = "customer"
    STAFF = "staff"
    ADMIN = "admin"


_ROLE_RANK = {Role.CUSTOMER: 0, Role.STAFF: 1, Role.ADMIN: 2}


@dataclass
class User:
    user_id: str
    name: str
    role: Role
    account_id: str | None = None  # 客户本人的账户
    can_approve: bool = False      # staff/admin 是否可审批

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role.value,
            "account_id": self.account_id,
            "can_approve": self.can_approve,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "User":
        return cls(
            d["user_id"],
            d["name"],
            Role(d["role"]),
            d.get("account_id"),
            d.get("can_approve", False),
        )


@dataclass
class PermissionResult:
    allowed: bool
    needs_approval: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "needs_approval": self.needs_approval,
            "reason": self.reason,
        }


def check_permission(user: User, spec: "ToolSpec", args: dict[str, Any]) -> PermissionResult:
    from banking_agent.tools.base import RiskLevel

    if _ROLE_RANK[user.role] < _ROLE_RANK[spec.required_role]:
        return PermissionResult(
            False, False, f"角色 {user.role.value} 无权使用工具 {spec.name}"
        )
    # 工具自定义授权（如：客户只能查自己的账户）
    if spec.authorize is not None:
        err = spec.authorize(user, args)
        if err:
            return PermissionResult(False, False, err)
    if spec.risk_level == RiskLevel.SENSITIVE:
        return PermissionResult(True, True, "敏感操作，需人工审批")
    return PermissionResult(True, False, "普通操作，允许直接执行")
