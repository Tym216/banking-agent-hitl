"""客户确认流程：草稿 → 挂起/恢复/修改 → 确认或取消（Case A/B 及边界）。"""

from __future__ import annotations

from banking_agent.bootstrap import DEMO_USERS
from banking_agent.tools import mock_bank


# ── Case A：转账 → 挂起 → 恢复 → 确认 → 审批 ──────────────

def test_case_a_transfer_suspend_resume_confirm(service):
    """转账草稿 → 查余额（挂起）→ 继续刚才的转账（恢复草稿）→ 确认 → 审批。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    tid = "cf-a-1"

    r = service.chat(tid, alice, "向账户 ACC-002 转账 300 元")
    assert "即将提交转账" in r["response"]
    assert "ACC-002" in r["response"]

    # 中途切走：查余额正常执行，草稿保持挂起
    r = service.chat(tid, alice, "查一下我的余额")
    assert "58200.5" in r["response"]
    assert mock_bank.TRANSACTIONS == []

    # 恢复草稿并再次展示
    r = service.chat(tid, alice, "继续刚才的转账")
    assert r["status"] == "completed"
    assert "即将提交转账" in r["response"]
    assert "300" in r["response"]

    # 确认 → 进入人工审批（未执行）
    r = service.chat(tid, alice, "确认")
    assert r["status"] == "pending_approval"
    assert r["approval_request"]["tool_args"]["amount"] == 300.0
    assert mock_bank.TRANSACTIONS == []


# ── Case B：工单 → 挂起 → 恢复并修改类别 → 确认创建 ──────────

def test_case_b_ticket_suspend_resume_modify_confirm(service):
    """工单草稿 → 查余额 → 继续工单并改类型为挂失 → 更新草稿 → 确认创建。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    tid = "cf-b-1"

    r = service.chat(tid, alice, "我要投诉，我的信用卡被多扣了费用")
    assert "即将创建工单" in r["response"]
    assert "投诉" in r["response"]

    # 中途切走
    r = service.chat(tid, alice, "查一下我的余额")
    assert "58200.5" in r["response"]
    assert mock_bank.TICKETS == []

    # 恢复工单并修改类别 → 草稿更新（摘要保留原内容）
    r = service.chat(tid, alice, "继续工单，把类型改成挂失")
    assert r["status"] == "completed"
    assert "即将创建工单" in r["response"]
    assert "挂失" in r["response"]
    assert "信用卡被多扣了费用" in r["response"]  # 摘要未丢

    # 确认 → 直接创建（create_ticket 无 staff 审批）
    r = service.chat(tid, alice, "确认")
    assert r["status"] == "completed"
    assert mock_bank.TICKETS
    ticket = mock_bank.TICKETS[-1]
    assert ticket["category"] == "card_loss"
    assert "信用卡被多扣了费用" in ticket["summary"]


# ── 取消路径 ─────────────────────────────────────────

def test_cancel_discards_draft(service):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    tid = "cf-c-1"

    r = service.chat(tid, alice, "向账户 ACC-002 转账 300 元")
    assert "即将提交转账" in r["response"]

    r = service.chat(tid, alice, "取消吧，不转了")
    assert "已取消" in r["response"]
    assert mock_bank.TRANSACTIONS == []

    # 取消后会话恢复正常，不会残留确认态
    r = service.chat(tid, alice, "查一下我的余额")
    assert "58200.5" in r["response"]


# ── 修改路径（转账改金额） ────────────────────────────

def test_modify_amount_then_confirm(service):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    tid = "cf-m-1"

    r = service.chat(tid, alice, "向账户 ACC-002 转账 300 元")
    assert "300" in r["response"]

    r = service.chat(tid, alice, "金额改成 500 元")
    assert "即将提交转账" in r["response"]
    assert "500.0" in r["response"]

    r = service.chat(tid, alice, "确认")
    assert r["status"] == "pending_approval"
    assert r["approval_request"]["tool_args"]["amount"] == 500.0
    assert mock_bank.TRANSACTIONS == []


# ── 其他意图切换不丢失草稿，且不触发执行 ──────────────────

def test_switch_intent_preserves_draft_and_no_side_effect(service):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    tid = "cf-s-1"

    r = service.chat(tid, alice, "我要投诉，ATM 吞卡了")
    assert "即将创建工单" in r["response"]
    assert mock_bank.TICKETS == []

    # 切换闲聊意图，工单不创建
    r = service.chat(tid, alice, "今天天气不错")
    assert mock_bank.TICKETS == []

    # 取消收尾
    r = service.chat(tid, alice, "算了，不用了")
    assert "已取消" in r["response"]
    assert mock_bank.TICKETS == []


# ── 工单频率限制（10 分钟 3 个） ──────────────────────

def _create_ticket_flow(service, tid: str, user, text: str) -> dict:
    r = service.chat(tid, user, text)
    assert "即将创建工单" in r["response"]
    return service.chat(tid, user, "确认")


def test_ticket_rate_limit_three_per_window(service):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    for i in range(3):
        r = _create_ticket_flow(service, f"cf-rl-{i}", alice, f"我要报障，第 {i} 个问题")
        assert r["status"] == "completed", f"第 {i+1} 个应创建成功"
    assert len(mock_bank.TICKETS) == 3

    # 第 4 个被限流，操作失败且不创建
    r = _create_ticket_flow(service, "cf-rl-3", alice, "我要报障，第四个问题")
    assert r["status"] == "completed"
    assert "过于频繁" in r["response"]
    assert len(mock_bank.TICKETS) == 3
