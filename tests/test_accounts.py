"""账号体系：密码哈希、登录校验、审批权限校验。"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from banking_agent.auth.accounts import (
    SEED_USERS,
    UserStore,
    hash_password,
    init_users,
    verify_password,
)
from banking_agent.auth.permissions import Role, User
from banking_agent.bootstrap import DEMO_USERS
from banking_agent.storage import connect


def _tmp_users(tmp_path: Path) -> UserStore:
    conn = connect(tmp_path / "agent.db")
    return init_users(conn, seed=True)


def test_password_hash_is_not_plaintext():
    h = hash_password("secret")
    assert "secret" not in h
    assert h.startswith("pbkdf2_sha256$")


def test_verify_password_correct_and_wrong():
    h = hash_password("secret")
    assert verify_password("secret", h) is True
    assert verify_password("wrong", h) is False
    assert verify_password("", h) is False


def test_same_password_different_hash():
    h1, h2 = hash_password("secret"), hash_password("secret")
    assert h1 != h2
    assert verify_password("secret", h1)
    assert verify_password("secret", h2)


def test_seed_users_login(tmp_path):
    store = _tmp_users(tmp_path)
    user = store.authenticate("alice", "alice123")
    assert user is not None
    assert user.user_id == "u_alice"
    assert user.role == Role.CUSTOMER
    assert user.account_id == "ACC-001"
    assert user.can_approve is False


def test_wrong_password_rejected(tmp_path):
    store = _tmp_users(tmp_path)
    assert store.authenticate("alice", "wrong") is None
    assert store.authenticate("nobody", "alice123") is None


def test_seed_staff_approve_permissions(tmp_path):
    store = _tmp_users(tmp_path)
    approver = store.authenticate("staff_approver", "staff123")
    viewer = store.authenticate("staff_viewer", "staff123")
    assert approver is not None and approver.can_approve is True
    assert viewer is not None and viewer.can_approve is False
    assert approver.role == Role.STAFF
    assert viewer.role == Role.STAFF


def test_approval_requires_staff_with_permission(service):
    """客户与无权限 staff 审批均被拒；有权限 staff 通过。"""
    alice = DEMO_USERS["u_alice"]
    staff_no_perm = User("u_x", "无权限柜员", Role.STAFF, can_approve=False)
    reply = service.chat("t-auth-1", alice, "向账户 ACC-002 转账 100 元")
    assert reply["status"] == "pending_approval"

    # 客户本人审批 → 拒绝
    reply = service.resolve_approval("t-auth-1", True, alice)
    assert reply["status"] == "error"
    assert "无权审批" in reply["message"]

    # staff 但无审批权限 → 拒绝
    reply = service.resolve_approval("t-auth-1", True, staff_no_perm)
    assert reply["status"] == "error"
    assert "审批权限" in reply["message"]

    # staff 有权限 → 通过
    reply = service.resolve_approval("t-auth-1", True, DEMO_USERS["u_staff"])
    assert reply["status"] == "completed"


def test_user_dict_roundtrip_with_can_approve():
    u = User("u1", "小明", Role.STAFF, can_approve=True)
    d = u.to_dict()
    assert d["can_approve"] is True
    restored = User.from_dict(d)
    assert restored.can_approve is True
    # 旧格式(无 can_approve 字段)兼容
    old = {"user_id": "u2", "name": "x", "role": "staff"}
    assert User.from_dict(old).can_approve is False


def test_login_audit_logged(tmp_path):
    conn = connect(tmp_path / "agent.db")
    store = init_users(conn, seed=True)
    store.authenticate("alice", "alice123")
    store.authenticate("alice", "wrong")
    rows = conn.execute(
        "SELECT entity_id, action FROM audit_logs ORDER BY id"
    ).fetchall()
    assert [(r["entity_id"], r["action"]) for r in rows] == [
        ("alice", "login_success"),
        ("alice", "login_failed"),
    ]


def test_login_updates_last_login_at(tmp_path):
    conn = connect(tmp_path / "agent.db")
    store = init_users(conn, seed=True)
    store.authenticate("alice", "alice123")
    ts = conn.execute(
        "SELECT last_login_at FROM users WHERE user_id = 'u_alice'"
    ).fetchone()["last_login_at"]
    assert ts is not None


def test_inactive_user_rejected(tmp_path):
    conn = connect(tmp_path / "agent.db")
    store = init_users(conn, seed=True)
    conn.execute("UPDATE users SET is_active = 0 WHERE user_id = 'u_alice'")
    conn.commit()
    assert store.authenticate("alice", "alice123") is None
    reason = conn.execute(
        "SELECT reason FROM audit_logs WHERE action = 'login_failed' ORDER BY id DESC LIMIT 1"
    ).fetchone()["reason"]
    assert reason == "账号已禁用"


def test_foreign_keys_reject_orphan_rows(tmp_path):
    conn = connect(tmp_path / "agent.db")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO messages (thread_id, role, content)"
            " VALUES ('no-such-thread', 'user', '孤儿消息')"
        )
