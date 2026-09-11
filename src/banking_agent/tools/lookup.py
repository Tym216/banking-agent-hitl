"""账户解析抽象：框架层按名字/账号解析账户，与具体 bank 适配器解耦。

GraphNodes 通过 AccountLookup 注入解析器（默认 MockBankAccountLookup），
nodes 不直接 import mock_bank。将来接真实核心系统/远程 MCP 时替换实现即可。
"""

from __future__ import annotations

from typing import Any, Protocol

from banking_agent.tools import mock_bank


class AccountLookup(Protocol):
    def lookup(self, term: str) -> dict[str, Any] | None:
        """按账号或持有人姓名解析账户；返回含 account_id/owner_user_id 的记录，失败 None。"""
        ...


class MockBankAccountLookup:
    """本地 mock bank 的账户解析（自含静态姓名注册表）。"""

    def lookup(self, term: str) -> dict[str, Any] | None:
        return mock_bank.lookup_account(term)


def default_account_lookup() -> AccountLookup:
    return MockBankAccountLookup()
