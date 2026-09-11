"""工具调用类：只读直接执行；敏感操作走审批，且审批前不产生副作用。"""

from banking_agent.bootstrap import DEMO_USERS
from banking_agent.tools import mock_bank


def test_balance_query_executes_directly(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-1", alice, "帮我查一下账户余额")
    assert reply["status"] == "completed"
    assert "58200.5" in reply["response"]


def _confirm_draft(service, thread_id, user, text: str):
    """需确认操作：草稿 → 回复确认，返回最终 reply。"""
    reply = service.chat(thread_id, user, text)
    assert "确认" in reply["response"], f"应先出现确认草稿: {reply}"
    return service.chat(thread_id, user, "确认")


def test_transfer_requires_approval_then_executes(service):
    alice = DEMO_USERS["u_alice"]
    reply = _confirm_draft(service, "t-tool-2", alice, "向账户 ACC-002 转账 500 元")
    assert reply["status"] == "pending_approval"
    assert reply["approval_request"]["tool_name"] == "submit_transaction"
    assert mock_bank.TRANSACTIONS == []  # 审批前不得执行

    reply = service.resolve_approval("t-tool-2", True, DEMO_USERS["u_staff"], "已核实")
    assert reply["status"] == "completed"
    assert len(mock_bank.TRANSACTIONS) == 1
    assert mock_bank.ACCOUNTS["ACC-002"]["balance"] == 1700.00


def test_transfer_rejected_by_approver(service):
    alice = DEMO_USERS["u_alice"]
    reply = _confirm_draft(service, "t-tool-3", alice, "向账户 ACC-002 转账 500 元")
    assert reply["status"] == "pending_approval"

    reply = service.resolve_approval("t-tool-3", False, DEMO_USERS["u_staff"], "无法核实收款方")
    assert reply["status"] == "completed"
    assert "未通过" in reply["response"]
    assert mock_bank.TRANSACTIONS == []
    assert mock_bank.ACCOUNTS["ACC-001"]["balance"] == 58200.50


def test_transfer_missing_amount_clarifies_then_completes(service):
    """缺参追问：框架层 pydantic 校验拦截（非 LLM 判断），补参后继续。"""
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-4", alice, "向账户 ACC-002 转账")
    assert reply["status"] == "completed"
    assert "还需要" in reply["response"] and "amount" in reply["response"]
    assert mock_bank.TRANSACTIONS == []

    reply = service.chat("t-tool-4", alice, "500 元")
    assert reply["status"] == "completed"
    assert "即将提交转账" in reply["response"]  # 先进入草稿确认

    reply = service.chat("t-tool-4", alice, "确认")
    assert reply["status"] == "pending_approval"

    reply = service.resolve_approval("t-tool-4", True, DEMO_USERS["u_staff"], "已核实")
    assert reply["status"] == "completed"
    assert len(mock_bank.TRANSACTIONS) == 1


def test_transfer_to_name_resolves_account(service):
    """收款方给姓名 → 框架层解析为账号 → 草稿显示 ACC-002 → 确认 → 审批。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-5", alice, "转账给 bob 300 元")
    assert reply["status"] == "completed"
    assert "即将提交转账" in reply["response"]
    assert "ACC-002" in reply["response"]  # 姓名已解析为账号
    assert mock_bank.TRANSACTIONS == []

    reply = service.chat("t-tool-5", alice, "确认")
    assert reply["status"] == "pending_approval"
    assert reply["approval_request"]["tool_args"]["to_account"] == "ACC-002"

    reply = service.resolve_approval("t-tool-5", True, DEMO_USERS["u_staff"], "已核实")
    assert reply["status"] == "completed"
    assert mock_bank.TRANSACTIONS and mock_bank.TRANSACTIONS[-1]["to_account"] == "ACC-002"


def test_transfer_from_other_name_refused(service):
    """from 指向他人姓名 → 拒绝且不泄露。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-6", alice, "从 bob 的账户转账 300 元给 ACC-003")
    assert reply["status"] == "completed"
    assert "无权" in reply["response"] and "拒绝" in reply["response"]
    assert "ACC-002" not in reply["response"]
    assert mock_bank.TRANSACTIONS == []


def test_transfer_to_unknown_asks_then_continue(service):
    """收款方无法解析 → 追问 → 用户补 ACC 账号后继续完整流程。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-7", alice, "转账给李四 200 元")
    assert "未识别" in reply["response"] or "ACC-" in reply["response"]
    assert mock_bank.TRANSACTIONS == []

    reply = service.chat("t-tool-7", alice, "收款账号是 ACC-003")
    assert reply["status"] == "completed"
    assert "即将提交转账" in reply["response"]
    assert "ACC-003" in reply["response"]

    reply = service.chat("t-tool-7", alice, "确认")
    assert reply["status"] == "pending_approval"
    assert reply["approval_request"]["tool_args"]["to_account"] == "ACC-003"


def test_transfer_from_own_by_name_ok(service):
    """from 用本人姓名 → 解析为本人，流程正常。"""
    mock_bank.reset()
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-8", alice, "从 alice 的账户转账给 bob 100 元")
    assert "即将提交转账" in reply["response"]
    reply = service.chat("t-tool-8", alice, "确认")
    assert reply["status"] == "pending_approval"
    args = reply["approval_request"]["tool_args"]
    assert args["from_account"] == "ACC-001"
    assert args["to_account"] == "ACC-002"
