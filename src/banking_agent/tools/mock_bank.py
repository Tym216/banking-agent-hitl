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
