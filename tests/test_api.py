"""API 端点测试（隔离临时 DB）：登录、待审批列表权限、审批执行。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BANKING_AGENT_STORAGE__DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv(
        "BANKING_AGENT_STORAGE__CHECKPOINT_DB_PATH", str(tmp_path / "ckpt.db")
    )
    monkeypatch.setenv("BANKING_AGENT_RAG__KB_DIR", "tests/data/mock_kb")
    monkeypatch.setenv("BANKING_AGENT_RAG__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("BANKING_AGENT_RAG__CHUNK_SIZE", "400")
    monkeypatch.setenv("BANKING_AGENT_LLM__PROVIDER", "mock")
    monkeypatch.setenv("BANKING_AGENT_EMBEDDING__PROVIDER", "mock")

    from banking_agent.api.app import app

    with TestClient(app) as c:
        yield c


def test_login_wrong_and_right(client):
    r = client.post("/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401
    r = client.post("/login", json={"username": "alice", "password": "alice123"})
    assert r.status_code == 200
    assert r.json()["user"]["role"] == "customer"


def _create_pending(client, thread_id: str = "api-ap-1") -> None:
    r = client.post(
        "/chat",
        json={"thread_id": thread_id, "message": "向账户 ACC-002 转账 100 元"},
        headers={"x-user-id": "u_alice"},
    )
    assert r.status_code == 200
    assert "确认" in r.json()["response"]  # 先出草稿
    r = client.post(
        "/chat",
        json={"thread_id": thread_id, "message": "确认"},
        headers={"x-user-id": "u_alice"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "pending_approval"


def test_pending_approvals_role_gate(client):
    _create_pending(client)

    # 客户 → 403
    r = client.get("/approvals/pending", headers={"x-user-id": "u_alice"})
    assert r.status_code == 403
    # staff 但无审批权限 → 403
    r = client.get("/approvals/pending", headers={"x-user-id": "u_staff_ro"})
    assert r.status_code == 403
    # 可审批 staff → 200 看到 1 条
    r = client.get("/approvals/pending", headers={"x-user-id": "u_staff"})
    assert r.status_code == 200
    rows = r.json()["approvals"]
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "api-ap-1"
    assert rows[0]["tool_name"] == "submit_transaction"
    assert rows[0]["requester_name"] == "Alice（客户）"


def test_approve_via_api_completes(client):
    _create_pending(client)

    # 客户审批 → 403
    r = client.post(
        "/approvals/api-ap-1/decision",
        json={"approved": True},
        headers={"x-user-id": "u_alice"},
    )
    assert r.status_code == 403

    # staff 审批 → completed，pending 清空
    r = client.post(
        "/approvals/api-ap-1/decision",
        json={"approved": True, "reason": "已核实"},
        headers={"x-user-id": "u_staff"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "completed"

    r = client.get("/approvals/pending", headers={"x-user-id": "u_staff"})
    assert r.json()["approvals"] == []
