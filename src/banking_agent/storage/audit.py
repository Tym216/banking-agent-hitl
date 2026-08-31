"""审计日志：对话、检索命中、工具调用链路、审批全流程落库。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


class AuditLogger:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        """底层连接，供账号体系(users 表)等复用同一数据库。"""
        return self._conn

    def ensure_conversation(self, thread_id: str, user_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO conversations (thread_id, user_id) VALUES (?, ?)",
            (thread_id, user_id),
        )
        self._conn.commit()

    def log_message(self, thread_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )
        self._conn.commit()

    def get_recent_messages(self, thread_id: str, limit: int = 8) -> list[dict[str, str]]:
        """按时间顺序返回该会话最近 limit 条消息，供多轮上下文使用。"""
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE thread_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def log_retrieval(self, thread_id: str, query: str, result: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO retrieval_logs (thread_id, query, decision, top_score, hits_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                query,
                result["decision"],
                result["top_score"],
                json.dumps(result["hits"], ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def log_tool_call(
        self,
        thread_id: str,
        tool_name: str,
        args: dict[str, Any],
        permission: dict[str, Any],
        approval_status: str = "not_required",
        result: dict[str, Any] | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO tool_calls"
            " (thread_id, tool_name, args_json, permission_json, approval_status, result_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                tool_name,
                json.dumps(args, ensure_ascii=False),
                json.dumps(permission, ensure_ascii=False),
                approval_status,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def update_tool_call(
        self,
        tool_call_id: int,
        approval_status: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if approval_status is not None:
            self._conn.execute(
                "UPDATE tool_calls SET approval_status = ? WHERE id = ?",
                (approval_status, tool_call_id),
            )
        if result is not None:
            self._conn.execute(
                "UPDATE tool_calls SET result_json = ? WHERE id = ?",
                (json.dumps(result, ensure_ascii=False), tool_call_id),
            )
        self._conn.commit()

    def request_approval(self, tool_call_id: int, thread_id: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO approvals (tool_call_id, thread_id) VALUES (?, ?)",
            (tool_call_id, thread_id),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def decide_approval(
        self, thread_id: str, approved: bool, approver: str, reason: str = ""
    ) -> None:
        self._conn.execute(
            "UPDATE approvals SET status = ?, approver = ?, reason = ?,"
            " decided_at = datetime('now')"
            " WHERE thread_id = ? AND status = 'pending'",
            ("approved" if approved else "rejected", approver, reason, thread_id),
        )
        self._conn.commit()

    def log_event(
        self,
        entity_type: str,
        entity_id: str | None,
        action: str,
        *,
        old_value: str | None = None,
        new_value: str | None = None,
        operator_id: str | None = None,
        operator_role: str | None = None,
        reason: str | None = None,
    ) -> None:
        """通用审计(append-only)：登录、角色变更、审批等敏感事件。"""
        self._conn.execute(
            "INSERT INTO audit_logs"
            " (entity_type, entity_id, action, old_value, new_value,"
            " operator_id, operator_role, reason)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entity_type, entity_id, action, old_value, new_value,
             operator_id, operator_role, reason),
        )
        self._conn.commit()
