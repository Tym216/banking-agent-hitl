"""只读查询工具：无需审批，但仍受角色与所有权校验。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from banking_agent.auth.permissions import Role, User
from banking_agent.tools import mock_bank
from banking_agent.tools.base import RiskLevel, ToolSpec


class BalanceParams(BaseModel):
    account_id: str = Field(description="账户号，如 ACC-001")


def _authorize_account_access(user: User, args: dict[str, Any]) -> str | None:
    """客户只能查询本人账户；staff 及以上可查任意账户。"""
    if user.role == Role.CUSTOMER and args.get("account_id") != user.account_id:
        return f"越权访问：客户 {user.user_id} 无权查询账户 {args.get('account_id')}"
    return None


def _get_balance(user: User, params: BaseModel) -> dict[str, Any]:
    assert isinstance(params, BalanceParams)
    return mock_bank.get_balance(params.account_id)


GET_ACCOUNT_BALANCE = ToolSpec(
    name="get_account_balance",
    description="查询账户余额",
    params_model=BalanceParams,
    risk_level=RiskLevel.READONLY,
    required_role=Role.CUSTOMER,
    handler=_get_balance,
    authorize=_authorize_account_access,
)
