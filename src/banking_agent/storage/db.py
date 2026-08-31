"""SQLite 存储：审计与业务数据表结构。

表结构近似标准 SQL，但含 SQLite 专属语法（datetime('now')、AUTOINCREMENT、
PRAGMA 外键开关），迁移 PostgreSQL 时需同步替换，见 AGENTS.md 数据规范。
LangGraph checkpoint 由 langgraph-checkpoint-sqlite 单独管理，不在此处。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,       -- PBKDF2-SHA256，格式见 auth/accounts.py
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('customer', 'staff', 'admin')),
    can_approve INTEGER NOT NULL DEFAULT 0,
    account_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES conversations(thread_id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 每次 RAG 检索的命中情况与三态判定
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES conversations(thread_id),
    query TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('answer', 'clarify', 'refuse')),
    top_score REAL NOT NULL,
    hits_json TEXT NOT NULL,          -- [{doc_id, chunk_id, score}, ...]
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 工具调用链路：参数、权限裁决、审批状态、执行结果
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES conversations(thread_id),
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
    thread_id TEXT NOT NULL REFERENCES conversations(thread_id),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected')),
    approver TEXT,
    reason TEXT,
    requested_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT
);

-- 通用审计(append-only)：登录、角色变更等敏感事件
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    action TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    operator_id TEXT,
    operator_role TEXT,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_thread ON tool_calls(thread_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_thread ON retrieval_logs(thread_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_approvals_tool_call ON approvals(tool_call_id);
"""

# 老库缺列时自动补齐（dev 阶段无正式迁移工具，新增列在这里登记）。
# 注意：ALTER TABLE ADD COLUMN 不允许非常量默认值，created_at 用可空列+回填。
_COLUMN_BACKFILLS: dict[str, list[tuple[str, str]]] = {
    "users": [
        ("created_at", "TEXT"),
        ("last_login_at", "TEXT"),
        ("is_active", "INTEGER NOT NULL DEFAULT 1"),
    ],
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    _backfill_columns(conn)
    return conn


def _backfill_columns(conn: sqlite3.Connection) -> None:
    """SQLite 的 CREATE TABLE IF NOT EXISTS 不会为新列改老表，这里逐列补。"""
    for table, columns in _COLUMN_BACKFILLS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        if "created_at" not in existing:
            conn.execute(
                f"UPDATE {table} SET created_at = datetime('now') WHERE created_at IS NULL"
            )
    conn.commit()
