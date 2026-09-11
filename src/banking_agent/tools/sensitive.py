"""敏感操作工具：注册为 SENSITIVE，框架强制走人工审批后才会调用 handler。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from banking_agent.auth.permissions import Role, User
from banking_agent.tools import mock_bank
from banking_agent.tools.base import RiskLevel, ToolSpec


class TicketParams(BaseModel):
    category: str = Field(description="工单类别：complaint / card_loss / general")
    summary: str = Field(min_length=2, max_length=500, description="问题摘要")


def _create_ticket(user: User, params: BaseModel) -> dict[str, Any]:
    assert isinstance(params, TicketParams)
    return mock_bank.create_ticket(user.user_id, params.category, params.summary)


class TransactionParams(BaseModel):
    from_account: str = Field(description="付款账户")
    to_account: str = Field(description="收款账户")
    amount: float = Field(gt=0, description="金额（元）")


def _authorize_transaction(user: User, args: dict[str, Any]) -> str | None:
    if user.role == Role.CUSTOMER and args.get("from_account") != user.account_id:
        return f"越权操作：客户 {user.user_id} 无权从账户 {args.get('from_account')} 转账"
    return None


def _submit_transaction(user: User, params: BaseModel) -> dict[str, Any]:
    assert isinstance(params, TransactionParams)
    return mock_bank.submit_transaction(
        params.from_account, params.to_account, params.amount
    )


CREATE_TICKET = ToolSpec(
    name="create_ticket",
    description="创建客服工单（投诉、挂失、报障等）",
    params_model=TicketParams,
    # 工单创建降级为"仅客户确认"：客户对草稿确认后直接创建，无需 staff 审批
    risk_level=RiskLevel.NORMAL,
    required_role=Role.CUSTOMER,
    handler=_create_ticket,
    requires_confirmation=True,
)

SUBMIT_TRANSACTION = ToolSpec(
    name="submit_transaction",
    description="提交转账交易",
    params_model=TransactionParams,
    risk_level=RiskLevel.SENSITIVE,
    required_role=Role.CUSTOMER,
    handler=_submit_transaction,
    authorize=_authorize_transaction,
    requires_confirmation=True,
)
