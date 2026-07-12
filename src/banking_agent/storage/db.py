"""SQLite 存储：审计与业务数据表结构。

标准 SQL，未来迁移 PostgreSQL 只需替换连接方式。
LangGraph checkpoint 由 langgraph-checkpoint-sqlite 单独管理，不在此处。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 每次 RAG 检索的命中情况与三态判定
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    query TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('answer', 'clarify', 'refuse')),
    top_score REAL NOT NULL,
    hits_json TEXT NOT NULL,          -- [{doc_id, chunk_id, score}, ...]
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 工具调用链路：参数、权限裁决、审批状态、执行结果
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    permission_json TEXT NOT NULL,    -- {allowed, needs_approval, reason}
    approval_status TEXT NOT NULL DEFAULT 'not_required'
        CHECK (approval_status IN ('not_required', 'pending', 'approved', 'rejected')),
    result_json TEXT,                 -- 执行结果；未执行为 NULL
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_call_id INTEGER NOT NULL REFERENCES tool_calls(id),
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected')),
    approver TEXT,
    reason TEXT,
    requested_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_thread ON tool_calls(thread_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_thread ON retrieval_logs(thread_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn
