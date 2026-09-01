"""LangGraph 状态定义。所有字段可 JSON 序列化，保证 checkpoint 持久化可用。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    thread_id: str
    user_input: str
    user: dict[str, Any]          # User.to_dict()
    history: list[dict[str, str]]  # 最近若干轮对话（不含本轮输入），由服务层从审计库加载
    intent: str                   # policy_qa | balance_query | create_ticket | transfer | chitchat
    retrieval: dict[str, Any]     # RetrievalResult.to_log_dict()
    # 缺参数被追问的工具调用：{"tool_name", "args", "missing"}。
    # 等参期间用户的补充/反问继续原流程；明确切换意图时清空。
    pending_tool: dict[str, Any]
    tool_name: str
    tool_args: dict[str, Any]
    permission: dict[str, Any]    # PermissionResult.to_dict()
    tool_call_id: int             # 审计表 tool_calls.id
    tool_result: dict[str, Any]
    approval: dict[str, Any]      # {approved, approver, reason}
    response: str
