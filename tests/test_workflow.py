"""工作流健壮性：审批挂起态、缺参等待态与非法恢复的边界行为。"""

from banking_agent.bootstrap import DEMO_USERS
from banking_agent.graph.nodes import GraphNodes
from banking_agent.tools import create_default_registry, mock_bank


def _confirm_draft(service, thread_id, user, text: str):
    """发送需确认的操作消息 → 草稿 → 回复确认，返回最终 reply。"""
    reply = service.chat(thread_id, user, text)
    assert "确认" in reply["response"], f"应先出现确认草稿: {reply}"
    return service.chat(thread_id, user, "确认")


def _is_param_ask(reply: dict) -> bool:
    """缺参/未识别追问话术（generic 或 unresolved 定制）。"""
    return "还需要" in reply["response"] or "未识别" in reply["response"]


def test_missing_param_waits_and_merges_followup(service):
    """缺参 → 反问不误触发其他工具 → 补充后合并参数 → 草稿 → 确认 → 审批。"""
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-pend-1", alice, "向账户转账500元")
    assert _is_param_ask(reply)

    reply = service.chat("t-pend-1", alice, "你是问我哪个账户？")
    assert _is_param_ask(reply)
    assert "58200.5" not in reply["response"]
    assert mock_bank.TRANSACTIONS == []

    # 补齐账户后进入草稿确认（而非直接审批）
    reply = service.chat("t-pend-1", alice, "转到 ACC-002 吧")
    assert reply["status"] == "completed"
    assert "即将提交转账" in reply["response"]

    reply = service.chat("t-pend-1", alice, "确认")
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


def test_prompt_env_injection_classify_and_extract():
    """classify 注入角色(minimal)；extract 注入完整身份(full)。"""
    captured: dict[str, str] = {}

    class SpyLLM:
        def chat(self, messages, **kwargs):
            captured["last_system"] = next(
                m["content"] for m in messages if m["role"] == "system"
            )
            if "[TASK:classify_intent]" in captured["last_system"]:
                return "balance_query"
            return "{}"

    nodes = GraphNodes(SpyLLM(), None, create_default_registry(), None, None)
    state_user = {"user_id": "u_alice", "name": "Alice（客户）",
                  "role": "customer", "account_id": "ACC-001"}
    state = {"user_input": "查一下我的余额", "user": state_user, "history": []}
    nodes.classify_intent(state)
    classify_system = captured["last_system"]
    assert "<user_min/>" not in classify_system
    assert "当前登录用户角色：客户（customer）" in classify_system
    assert "ACC-001" not in classify_system  # classify 只注入角色

    nodes.prepare_tool({**state, "intent": "balance_query"})
    extract_system = captured["last_system"]
    assert "当前登录用户：Alice（客户）" in extract_system
    assert "u_alice" in extract_system and "ACC-001" in extract_system


def test_prompt_env_injection_answer_datetime():
    """answer 注入本地日期时间。"""
    from banking_agent.graph.nodes import _ANSWER_PROMPT, _with_history

    messages = _with_history(
        _ANSWER_PROMPT.format(context="资料内容"),
        {"user_input": "年费多少", "user": {}, "history": []},
    )
    system = messages[0]["content"]
    assert "<datetime/>" not in system
    assert "今天是 " in system and "年" in system


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
    reply = _confirm_draft(service, "t-wf-1", alice, "向账户 ACC-002 转账 300 元")
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
