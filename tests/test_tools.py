"""工具调用类：只读直接执行；敏感操作走审批，且审批前不产生副作用。"""

from banking_agent.bootstrap import DEMO_USERS
from banking_agent.tools import mock_bank


def test_balance_query_executes_directly(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-1", alice, "帮我查一下账户余额")
    assert reply["status"] == "completed"
    assert "58200.5" in reply["response"]


def test_transfer_requires_approval_then_executes(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-2", alice, "向账户 ACC-002 转账 500 元")
    assert reply["status"] == "pending_approval"
    assert reply["approval_request"]["tool_name"] == "submit_transaction"
    assert mock_bank.TRANSACTIONS == []  # 审批前不得执行

    reply = service.resolve_approval("t-tool-2", True, DEMO_USERS["u_staff"], "已核实")
    assert reply["status"] == "completed"
    assert len(mock_bank.TRANSACTIONS) == 1
    assert mock_bank.ACCOUNTS["ACC-002"]["balance"] == 1700.00


def test_transfer_rejected_by_approver(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-tool-3", alice, "向账户 ACC-002 转账 500 元")
    assert reply["status"] == "pending_approval"

    reply = service.resolve_approval("t-tool-3", False, DEMO_USERS["u_staff"], "无法核实收款方")
    assert reply["status"] == "completed"
    assert "未通过" in reply["response"]
    assert mock_bank.TRANSACTIONS == []
    assert mock_bank.ACCOUNTS["ACC-001"]["balance"] == 58200.50
