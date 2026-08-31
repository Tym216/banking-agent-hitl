"""工作流健壮性：审批挂起态、缺参等待态与非法恢复的边界行为。"""

from banking_agent.bootstrap import DEMO_USERS
from banking_agent.graph.nodes import GraphNodes
from banking_agent.tools import create_default_registry, mock_bank


def test_missing_param_waits_and_merges_followup(service):
    """缺参 → 反问不误触发其他工具 → 补充后合并参数继续原流程。"""
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-pend-1", alice, "向账户转账500元")
    assert "还需要" in reply["response"]

    reply = service.chat("t-pend-1", alice, "你是问我哪个账户？")
    assert "还需要" in reply["response"]
    assert "58200.5" not in reply["response"]
    assert mock_bank.TRANSACTIONS == []

    reply = service.chat("t-pend-1", alice, "转到 ACC-002 吧")
    assert reply["status"] == "pending_approval"
    args = reply["approval_request"]["tool_args"]
    assert args["to_account"] == "ACC-002"
    assert args["amount"] == 500.0


def test_pending_tool_dropped_on_intent_switch(service):
    alice = DEMO_USERS["u_alice"]
    service.chat("t-pend-2", alice, "向账户转账500元")
    reply = service.chat("t-pend-2", alice, "查一下我的余额")
    assert "58200.5" in reply["response"]
    assert mock_bank.TRANSACTIONS == []


def test_param_extraction_only_sees_user_messages():
    """助手历史回复中的参数值不得进入抽取上下文。"""
    captured: dict = {}

    class SpyLLM:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            return "{}"

    nodes = GraphNodes(SpyLLM(), None, create_default_registry(), None, None)
    nodes.prepare_tool(
        {
            "intent": "balance_query",
            "user_input": "帮我查账户 ACC-003 的余额",
            "user": {"role": "customer", "account_id": "ACC-001"},
            "history": [
                {"role": "user", "content": "查一下我的账户余额"},
                {"role": "assistant", "content": '{"account_id": "ACC-001", "balance": 58200.5}'},
            ],
        }
    )
    roles = {m["role"] for m in captured["messages"]}
    assert "assistant" not in roles
    assert all("ACC-001" not in m["content"] for m in captured["messages"] if m["role"] != "system")


def test_ambiguous_query_clarify_then_classify_with_context(service):
    """模糊的"查账户"先追问；用户说明"余额"后结合上下文正确路由。"""
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-ambig-1", alice, "帮我查一下我的账户")
    assert "58200.5" not in reply["response"]  # 不得默认执行余额查询

    reply = service.chat("t-ambig-1", alice, "我想查余额")
    assert reply["status"] == "completed"
    assert "58200.5" in reply["response"]


def test_resolve_approval_without_pending_returns_error(service):
    staff = DEMO_USERS["u_staff"]
    reply = service.resolve_approval("t-wf-nothing", True, staff)
    assert reply["status"] == "error"


def test_new_message_while_pending_is_not_swallowed(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-wf-1", alice, "向账户 ACC-002 转账 300 元")
    assert reply["status"] == "pending_approval"

    # 审批未决时继续对话：应重新提示待审批，且不触发任何执行
    reply = service.chat("t-wf-1", alice, "查一下我的余额")
    assert reply["status"] == "pending_approval"
    assert "审批" in reply["note"]
    assert mock_bank.TRANSACTIONS == []

    # 完成审批后会话恢复正常
    reply = service.resolve_approval("t-wf-1", True, DEMO_USERS["u_staff"], "已核实")
    assert reply["status"] == "completed"
    assert len(mock_bank.TRANSACTIONS) == 1
    reply = service.chat("t-wf-1", alice, "查一下我的余额")
    assert reply["status"] == "completed"
    assert "57900.5" in reply["response"]  # 58200.5 - 300
