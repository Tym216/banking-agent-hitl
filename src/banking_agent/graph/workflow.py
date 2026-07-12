"""图编排与对外服务封装。

图结构：
  classify_intent ─┬─ policy_qa ──────────────────────────────► END
                   ├─ prepare_tool → check_permission ─┬─(拒绝/缺参)► END
                   │                                   ├─(只读)──► execute_tool ► END
                   │                                   └─(敏感)──► approval ─┬─(通过)► execute_tool
                   └─ chitchat ───────────────────────► END      └─(驳回)────────► END

approval 节点内的 interrupt() + SqliteSaver checkpointer 实现中断/恢复：
进程重启后凭 thread_id 依然可以恢复待审批的工作流。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from banking_agent.auth.permissions import User
from banking_agent.config import AppConfig
from banking_agent.graph.nodes import GraphNodes
from banking_agent.graph.state import AgentState
from banking_agent.llm.base import LLMClient
from banking_agent.rag.retriever import Retriever
from banking_agent.storage.audit import AuditLogger
from banking_agent.tools.base import ToolRegistry


def build_graph(nodes: GraphNodes, checkpointer: Any):
    g = StateGraph(AgentState)
    g.add_node("classify_intent", nodes.classify_intent)
    g.add_node("policy_qa", nodes.policy_qa)
    g.add_node("prepare_tool", nodes.prepare_tool)
    g.add_node("check_permission", nodes.check_permission)
    g.add_node("approval", nodes.approval)
    g.add_node("execute_tool", nodes.execute_tool)
    g.add_node("chitchat", nodes.chitchat)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        lambda s: s["intent"],
        {
            "policy_qa": "policy_qa",
            "balance_query": "prepare_tool",
            "create_ticket": "prepare_tool",
            "transfer": "prepare_tool",
            "chitchat": "chitchat",
        },
    )
    g.add_edge("prepare_tool", "check_permission")

    def _route_permission(s: AgentState) -> str:
        perm = s["permission"]
        if not perm["allowed"]:
            return "denied"
        return "approval" if perm["needs_approval"] else "execute"

    g.add_conditional_edges(
        "check_permission",
        _route_permission,
        {"denied": END, "approval": "approval", "execute": "execute_tool"},
    )
    g.add_conditional_edges(
        "approval",
        lambda s: "execute" if s["approval"]["approved"] else "rejected",
        {"execute": "execute_tool", "rejected": END},
    )
    g.add_edge("policy_qa", END)
    g.add_edge("execute_tool", END)
    g.add_edge("chitchat", END)
    return g.compile(checkpointer=checkpointer)


class AgentService:
    """对 CLI / FastAPI 暴露的统一入口：chat() 与 resolve_approval()。"""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        retriever: Retriever,
        registry: ToolRegistry,
        audit: AuditLogger,
    ) -> None:
        ckpt_path = config.resolve_path(config.storage.checkpoint_db_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(ckpt_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(conn)
        nodes = GraphNodes(llm, retriever, registry, audit, config)
        self._graph = build_graph(nodes, self._checkpointer)
        self._audit = audit

    def chat(self, thread_id: str, user: User, text: str) -> dict[str, Any]:
        # 会话有待审批操作时不接受新输入，否则会触发 interrupt 节点重放、吞掉消息
        pending = self._pending_request(thread_id)
        if pending is not None:
            return {
                "status": "pending_approval",
                "approval_request": pending,
                "note": "该会话有操作正在等待人工审批，请先完成审批再继续对话。",
            }
        self._audit.ensure_conversation(thread_id, user.user_id)
        history = self._audit.get_recent_messages(thread_id, limit=8)
        self._audit.log_message(thread_id, "user", text)
        result = self._graph.invoke(
            {
                "thread_id": thread_id,
                "user_input": text,
                "user": user.to_dict(),
                "history": history,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return self._to_reply(thread_id, result)

    def resolve_approval(
        self, thread_id: str, approved: bool, approver: str, reason: str = ""
    ) -> dict[str, Any]:
        if self._pending_request(thread_id) is None:
            return {"status": "error", "message": f"会话 {thread_id} 没有待审批的操作"}
        result = self._graph.invoke(
            Command(resume={"approved": approved, "approver": approver, "reason": reason}),
            config={"configurable": {"thread_id": thread_id}},
        )
        return self._to_reply(thread_id, result)

    def _pending_request(self, thread_id: str) -> dict[str, Any] | None:
        """返回该会话待审批请求的 payload；无待审批返回 None。"""
        snapshot = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    def _to_reply(self, thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value
            return {"status": "pending_approval", "approval_request": payload}
        response = result.get("response", "")
        self._audit.log_message(thread_id, "assistant", response)
        return {"status": "completed", "response": response}
