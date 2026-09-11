"""CLI 多会话与审批流程测试：客户不弹审批、惰性开新会话、staff 审批、回看结果。"""

from __future__ import annotations

from banking_agent.auth.permissions import Role, User
from banking_agent.bootstrap import DEMO_USERS
from banking_agent.cli import Session
from banking_agent.tools import mock_bank


def _run(service, user, inputs: list[str], capsys) -> str:
    """用注入的输入序列跑 Session，返回全部输出文本。"""
    it = iter(inputs)

    def fake_input(prompt: str = "") -> str:
        return next(it)

    Session(service, user, get_input=fake_input).run()
    return capsys.readouterr().out


def _new_user(user_id: str, name: str, role: Role, can_approve: bool = False) -> User:
    return User(user_id, name, role, can_approve=can_approve)


def test_customer_transfer_no_prompt_then_auto_new_session(service, capsys):
    """客户转账 → 确认草稿 → 不弹 y/n、提示等待 → 下一条消息自动开新会话。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    out = _run(service, alice, [
        "向账户 ACC-002 转账 300 元",   # → 草稿
        "确认",                         # → pending，提示等待（无 y/n）
        "查一下我的余额",               # 触发惰性换会话
        "q",
    ], capsys)

    # 1. 先出草稿，再提示等待，且客户从不被询问"批准该操作"
    assert "即将提交转账" in out
    assert "已提交审批" in out
    assert "等待工作人员处理" in out
    assert "批准该操作" not in out
    # 2. 自动开了新会话并正常应答余额
    assert "已开启新会话" in out
    assert "58200.5" in out
    assert mock_bank.TRANSACTIONS == []  # 审批前不执行


def test_staff_sees_and_approves_pending(service, capsys):
    """staff 登录后 /approvals 能看到客户的待审批，/approve y 后完成。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    out = _run(service, alice, [
        "向账户 ACC-002 转账 300 元",
        "确认",
        "q",
    ], capsys)
    assert "已提交审批" in out

    staff = _new_user("u_staff", "柜员王", Role.STAFF, can_approve=True)
    out2 = _run(service, staff, [
        "/approvals",
        "/approve 1 y 已核实收款方",
        "q",
    ], capsys)
    assert "待审批列表" in out2
    assert "submit_transaction" in out2
    assert "已批准" in out2
    # 原因入库
    reason = service._audit.conn.execute(
        "SELECT reason FROM approvals WHERE status='approved' LIMIT 1"
    ).fetchone()[0]
    assert reason == "已核实收款方"
    assert mock_bank.TRANSACTIONS  # 批准后真实执行


def test_staff_can_reject(service, capsys):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    _run(service, alice, ["向账户 ACC-002 转账 300 元", "确认", "q"], capsys)
    staff = _new_user("u_staff", "柜员王", Role.STAFF, can_approve=True)
    out = _run(service, staff, ["/approve 1 n 收款方存疑", "q"], capsys)
    assert "已拒绝" in out
    assert mock_bank.TRANSACTIONS == []


def test_customer_switch_back_sees_result(service, capsys):
    """客户 /switch 回原会话能看到审批结果（已批准的操作消息）。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    out1 = _run(service, alice, ["向账户 ACC-002 转账 300 元", "确认", "q"], capsys)
    assert "已提交审批" in out1
    # 取出 alice 的第一个会话 id
    first_tid = service._audit.conn.execute(
        "SELECT thread_id FROM conversations WHERE user_id = 'u_alice'"
        " ORDER BY started_at LIMIT 1"
    ).fetchone()[0]

    staff = _new_user("u_staff", "柜员王", Role.STAFF, can_approve=True)
    _run(service, staff, ["/approve 1 y", "q"], capsys)

    out2 = _run(service, alice, [
        f"/switch {first_tid}",
        "/status",
        "q",
    ], capsys)
    assert "已切换到会话" in out2
    assert "已批准" in out2          # /status 显示审批结果
    assert "操作已完成" in out2      # 最近消息含执行结果
    assert mock_bank.TRANSACTIONS


def test_customer_cannot_view_pending_approvals(service, capsys):
    """客户 /approvals 被拒。"""
    alice = DEMO_USERS["u_alice"]
    out = _run(service, alice, ["/approvals", "q"], capsys)
    assert "无权查看" in out


def test_staff_viewer_cannot_approve(service, capsys):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    _run(service, alice, ["向账户 ACC-002 转账 300 元", "确认", "q"], capsys)
    ro = _new_user("u_staff_ro", "柜员李", Role.STAFF, can_approve=False)
    out = _run(service, ro, ["/approvals", "q"], capsys)
    assert "无权查看" in out


# ── audit 查询函数 ─────────────────────────────────

def test_list_user_threads_only_own(service):
    alice = DEMO_USERS["u_alice"]
    service.chat("t-list-1", alice, "查一下我的余额")
    service.chat("t-list-2", alice, "信用卡年费政策")
    rows = service._audit.list_user_threads("u_alice")
    tids = {r["thread_id"] for r in rows}
    assert {"t-list-1", "t-list-2"} <= tids
    # 标题 = 首条用户消息；limit 生效
    by_id = {r["thread_id"]: r for r in rows}
    assert by_id["t-list-1"]["first_msg"] == "查一下我的余额"
    assert len(service._audit.list_user_threads("u_alice", limit=1)) == 1


def test_thread_approval_status_records_decision(service, capsys):
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    _run(service, alice, ["向账户 ACC-002 转账 300 元", "确认", "q"], capsys)
    staff = _new_user("u_staff", "柜员王", Role.STAFF, can_approve=True)
    _run(service, staff, ["/approve 1 y ok", "q"], capsys)
    tid = service._audit.conn.execute(
        "SELECT thread_id FROM approvals WHERE status='approved' LIMIT 1"
    ).fetchone()[0]
    recs = service._audit.thread_approval_status(tid)
    assert recs and recs[0]["status"] == "approved"
    assert recs[0]["approver"] == "u_staff"
