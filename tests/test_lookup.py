"""mock_bank.lookup_account：账号/姓名精确小写匹配，禁子串模糊。"""

from __future__ import annotations

from banking_agent.tools import mock_bank


def test_account_exact_match():
    r = mock_bank.lookup_account("ACC-002")
    assert r and r["account_id"] == "ACC-002"
    assert r["owner_user_id"] == "u_bob"
    # 小写账号同样命中
    assert mock_bank.lookup_account("acc-002")["account_id"] == "ACC-002"


def test_name_case_insensitive():
    assert mock_bank.lookup_account("bob")["account_id"] == "ACC-002"
    assert mock_bank.lookup_account("BOB")["account_id"] == "ACC-002"
    assert mock_bank.lookup_account("Alice（客户）")["account_id"] == "ACC-001"
    assert mock_bank.lookup_account("alice")["account_id"] == "ACC-001"
    assert mock_bank.lookup_account("Carol")["account_id"] == "ACC-003"


def test_no_substring_fuzz():
    """精确匹配：bob 不命中 bobby；前后缀不模糊。"""
    assert mock_bank.lookup_account("bobby") is None
    assert mock_bank.lookup_account("alices") is None
    assert mock_bank.lookup_account("abob") is None


def test_unknown_returns_none():
    assert mock_bank.lookup_account("张三") is None
    assert mock_bank.lookup_account("ACC-999") is None
    assert mock_bank.lookup_account("") is None
    assert mock_bank.lookup_account("  ") is None
    assert mock_bank.lookup_account(None) is None


def test_lookup_result_has_no_existence_ambiguity():
    """记录含持有者信息；每名字唯一映射。"""
    r = mock_bank.lookup_account("bob")
    assert r["name"] == "Bob（客户）"
    assert "balance" in r and "currency" in r
