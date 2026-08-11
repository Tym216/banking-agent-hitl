"""模拟银行核心系统：内存账本，后续可替换为真实业务系统适配器。"""

from __future__ import annotations

import itertools
from typing import Any

ACCOUNTS: dict[str, dict[str, Any]] = {
    "ACC-001": {"owner_user_id": "u_alice", "balance": 58200.50, "currency": "CNY"},
    "ACC-002": {"owner_user_id": "u_bob", "balance": 1200.00, "currency": "CNY"},
    "ACC-003": {"owner_user_id": "u_carol", "balance": 990000.00, "currency": "CNY"},
}

TICKETS: list[dict[str, Any]] = []
TRANSACTIONS: list[dict[str, Any]] = []

_ticket_seq = itertools.count(1)
_txn_seq = itertools.count(1)


def next_ticket_id() -> str:
    return f"TKT-{next(_ticket_seq):05d}"


def next_txn_id() -> str:
    return f"TXN-{next(_txn_seq):05d}"


# ---------- 业务操作(单一实现,供本地工具 handler 与 MCP server 共用) ----------

def get_balance(account_id: str) -> dict[str, Any]:
    acct = ACCOUNTS.get(account_id)
    if acct is None:
        return {"ok": False, "error": f"账户 {account_id} 不存在"}
    return {
        "ok": True,
        "account_id": account_id,
        "balance": acct["balance"],
        "currency": acct["currency"],
    }


def create_ticket(user_id: str, category: str, summary: str) -> dict[str, Any]:
    ticket = {
        "ticket_id": next_ticket_id(),
        "user_id": user_id,
        "category": category,
        "summary": summary,
        "status": "open",
    }
    TICKETS.append(ticket)
    return {"ok": True, **ticket}


def submit_transaction(from_account: str, to_account: str, amount: float) -> dict[str, Any]:
    src = ACCOUNTS.get(from_account)
    dst = ACCOUNTS.get(to_account)
    if src is None or dst is None:
        return {"ok": False, "error": "付款或收款账户不存在"}
    if src["balance"] < amount:
        return {"ok": False, "error": "余额不足"}
    src["balance"] -= amount
    dst["balance"] += amount
    txn = {
        "txn_id": next_txn_id(),
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
    }
    TRANSACTIONS.append(txn)
    return {"ok": True, **txn}


def reset() -> None:
    """测试用：还原账本。"""
    global _ticket_seq, _txn_seq
    TICKETS.clear()
    TRANSACTIONS.clear()
    ACCOUNTS["ACC-001"]["balance"] = 58200.50
    ACCOUNTS["ACC-002"]["balance"] = 1200.00
    ACCOUNTS["ACC-003"]["balance"] = 990000.00
    _ticket_seq = itertools.count(1)
    _txn_seq = itertools.count(1)
