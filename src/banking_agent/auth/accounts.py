"""账号体系：PBKDF2 密码哈希 + 用户存储 + 种子数据。

密码从不存明文；哈希算法封装在本模块，将来换 argon2/bcrypt 只改这里。
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3

from banking_agent.auth.permissions import Role, User

_PBKDF2_ITERATIONS = 100_000
_SALT_BYTES = 16
_HASH_NAME = "sha256"

SEED_USERS: list[dict] = [
    # user_id, username, password, name, role, can_approve, account_id
    ("u_alice", "alice", "alice123", "Alice（客户）", Role.CUSTOMER, False, "ACC-001"),
    ("u_bob", "bob", "bob123", "Bob（客户）", Role.CUSTOMER, False, "ACC-002"),
    ("u_carol", "carol", "carol123", "Carol（客户）", Role.CUSTOMER, False, "ACC-003"),
    ("u_zhou", "zhou", "zhou123", "周先生（客户）", Role.CUSTOMER, False, "ACC-004"),
    ("u_liang", "liang", "liang123", "梁小姐（客户）", Role.CUSTOMER, False, "ACC-005"),
    ("u_zheng", "zheng", "zheng123", "郑太太（客户）", Role.CUSTOMER, False, "ACC-006"),
    ("u_staff", "staff_approver", "staff123", "柜员王（可审批）", Role.STAFF, True, None),
    ("u_staff_ro", "staff_viewer", "staff123", "柜员李（不可审批）", Role.STAFF, False, None),
]


def hash_password(password: str) -> str:
    """返回 "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(_HASH_NAME, password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_{_HASH_NAME}${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, hash_hex = stored.split("$")
        name = algo.removeprefix("pbkdf2_")
        dk = hashlib.pbkdf2_hmac(
            name, password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return secrets.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


class UserStore:
    """从 users 表加载/校验用户。所有账号查询的入口。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_by_username(self, username: str) -> User | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return self._row_to_user(row)

    def get_by_id(self, user_id: str) -> User | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return self._row_to_user(row)

    def authenticate(self, username: str, password: str) -> User | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        ok = row is not None and bool(row["is_active"]) and verify_password(
            password, row["password_hash"]
        )
        reason = (
            "" if ok
            else ("账号已禁用" if row is not None and not row["is_active"] else "密码错误或用户不存在")
        )
        with self._conn:  # last_login_at 与审计日志同事务，避免部分写入
            if ok:
                self._conn.execute(
                    "UPDATE users SET last_login_at = datetime('now')"
                    " WHERE user_id = ?",
                    (row["user_id"],),
                )
            self._log_login(self._conn, username, ok, reason)
        return self._row_to_user(row) if ok else None

    @staticmethod
    def _log_login(conn: sqlite3.Connection, username: str, ok: bool, reason: str) -> None:
        conn.execute(
            "INSERT INTO audit_logs (entity_type, entity_id, action, new_value,"
            " operator_id, operator_role, reason)"
            " VALUES ('user', ?, ?, ?, ?, 'customer', ?)",
            (username, "login_success" if ok else "login_failed",
             "ok" if ok else "denied", username, reason),
        )

    @staticmethod
    def _row_to_user(row: sqlite3.Row | None) -> User | None:
        if row is None:
            return None
        return User(
            user_id=row["user_id"],
            name=row["name"],
            role=Role(row["role"]),
            account_id=row["account_id"],
            can_approve=bool(row["can_approve"]),
        )


def init_users(conn: sqlite3.Connection, seed: bool = True) -> UserStore:
    """初始化 users 表 (建表由 storage/db.py 的 schema 完成), 可选写入种子用户。"""
    if seed:
        for user_id, username, password, name, role, can_approve, account_id in SEED_USERS:
            conn.execute(
                """
                INSERT INTO users (user_id, username, password_hash, name, role, can_approve, account_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (user_id, username, hash_password(password), name, role.value,
                 int(can_approve), account_id),
            )
        conn.commit()
    return UserStore(conn)
