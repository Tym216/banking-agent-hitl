"""模拟银行核心系统：内存账本，后续可替换为真实业务系统适配器。"""

from __future__ import annotations

import itertools
from typing import Any

ACCOUNTS: dict[str, dict[str, Any]] = {
    "ACC-001": {"owner_user_id": "u_alice", "balance": 58200.50, "currency": "CNY"},
    "ACC-002": {"owner_user_id": "u_bob", "balance": 1200.00, "currency": "CNY"},
    "ACC-003": {"owner_user_id": "u_carol", "balance": 990000.00, "currency": "CNY"},
    "ACC-004": {"owner_user_id": "u_zhou", "balance": 35600.00, "currency": "CNY"},
    "ACC-005": {"owner_user_id": "u_liang", "balance": 8800.50, "currency": "CNY"},
    "ACC-006": {"owner_user_id": "u_zheng", "balance": 152000.00, "currency": "CNY"},
}

# 账户持有人的登记信息（与 users 表种子一致；bank 侧自洽，MCP 独立进程同样可用）
ACCOUNT_OWNERS: dict[str, dict[str, str]] = {
    "ACC-001": {"user_id": "u_alice", "name": "Alice（客户）"},
    "ACC-002": {"user_id": "u_bob", "name": "Bob（客户）"},
    "ACC-003": {"user_id": "u_carol", "name": "Carol（客户）"},
    "ACC-004": {"user_id": "u_zhou", "name": "周先生（客户）"},
    "ACC-005": {"user_id": "u_liang", "name": "梁小姐（客户）"},
    "ACC-006": {"user_id": "u_zheng", "name": "郑太太（客户）"},
}

# reset() 基线：初始余额快照（新增账户只改这里）
_INITIAL_BALANCES: dict[str, float] = {
    "ACC-001": 58200.50,
    "ACC-002": 1200.00,
    "ACC-003": 990000.00,
    "ACC-004": 35600.00,
    "ACC-005": 8800.50,
    "ACC-006": 152000.00,
}


def _clean_key(text: str) -> str:
    """清洗姓名键：去括号后缀/空白，小写。仅用于精确匹配，不做模糊。"""
    import re

    return re.sub(r"[（(].*?[)）]", "", text).strip().lower()


# 姓名别名 → 账号（精确小写匹配用，不做子串模糊）
_NAME_ALIASES: dict[str, str] = {}
for _acc, _owner in ACCOUNT_OWNERS.items():
    for _alias in (_owner["name"], _owner["name"].split("（")[0], _owner["user_id"][2:]):
        _NAME_ALIASES.setdefault(_clean_key(_alias), _acc)


def lookup_account(name_or_account: str) -> dict[str, Any] | None:
    """按账号（ACC-xxx 精确）或持有人姓名（清洗后精确、大小写不敏感）查账户。

    找不到返回 None。不做子串模糊匹配（"bob" 不会命中 "bobby"）。
    """
    term = (name_or_account or "").strip()
    if not term:
        return None
    account_id = term.upper() if term.upper().startswith("ACC-") else None
    if account_id is None or account_id not in ACCOUNTS:
        account_id = _NAME_ALIASES.get(_clean_key(term))
    if account_id is None or account_id not in ACCOUNTS:
        return None
    acct = ACCOUNTS[account_id]
    owner = ACCOUNT_OWNERS.get(account_id, {})
    return {
        "account_id": account_id,
        "owner_user_id": acct["owner_user_id"],
        "name": owner.get("name", ""),
        "balance": acct["balance"],
        "currency": acct["currency"],
    }

TICKETS: list[dict[str, Any]] = []
TRANSACTIONS: list[dict[str, Any]] = []

_ticket_seq = itertools.count(1)
_txn_seq = itertools.count(1)

# 工单创建频率限制：10 分钟内最多 3 个（按 user_id）
_TICKET_WINDOW_SECONDS = 600
_TICKET_MAX_PER_WINDOW = 3
_ticket_timestamps: dict[str, list[float]] = {}


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
    import time

    now = time.time()
    recent = [t for t in _ticket_timestamps.get(user_id, [])
              if now - t < _TICKET_WINDOW_SECONDS]
    if len(recent) >= _TICKET_MAX_PER_WINDOW:
        return {
            "ok": False,
            "error": "工单创建过于频繁，请在 10 分钟后再试（10 分钟内最多 3 个工单）。",
        }
    recent.append(now)
    _ticket_timestamps[user_id] = recent

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
    """测试用：还原账本（余额按基线快照，含新增账户）。"""
    global _ticket_seq, _txn_seq, _ticket_timestamps
    TICKETS.clear()
    TRANSACTIONS.clear()
    for account_id, balance in _INITIAL_BALANCES.items():
        if account_id in ACCOUNTS:
            ACCOUNTS[account_id]["balance"] = balance
    _ticket_seq = itertools.count(1)
    _txn_seq = itertools.count(1)
    _ticket_timestamps = {}
