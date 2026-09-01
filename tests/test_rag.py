"""政策问答路径：检索空→拒答；检索命中→LLM 自主决定回答或追问。

mock embedder 下"命中⟺问题与文档有 bigram 重合"，因此 LLM 自主追问分支
无法通过图级测试确定性触发，改为对 mock._answer 做单元测试（见文件末尾）。
"""

from banking_agent.bootstrap import DEMO_USERS
from banking_agent.llm.mock import MockLLMClient


def test_policy_question_answered(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-rag-1", alice, "信用卡年费政策是怎么收取的？")
    assert reply["status"] == "completed"
    assert "年费" in reply["response"]
    # 高置信回答必须带知识库来源，而不是模型裸答
    assert "知识库" in reply["response"] or "政策资料" in reply["response"]


def test_unknown_topic_is_refused(service):
    """检索为空（min_score 粗滤后无候选）→ 框架层拒答，不依赖 LLM。"""
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-rag-2", alice, "请问怎样申请办理火星移民贷款业务呀？")
    assert reply["status"] == "completed"
    assert "无法回答" in reply["response"] or "没有找到" in reply["response"]


def test_llm_answers_when_context_sufficient(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-rag-3", alice, "信用卡取现手续费是多少")
    assert reply["status"] == "completed"
    assert "政策资料" in reply["response"] or "知识库" in reply["response"]


# ── LLM 自主追问分支（单元级） ─────────────────────────

_MOCK = MockLLMClient()


def _answer_prompt(context: str, question: str) -> list[dict]:
    from banking_agent.graph.nodes import _ANSWER_PROMPT, _with_history

    return _with_history(_ANSWER_PROMPT.format(context=context),
                         {"user_input": question, "history": []})


def test_mock_llm_clarifies_when_context_has_no_overlap():
    """问题与上下文无任何重合 → 追问而非强行回答。"""
    prompt = _answer_prompt("信用卡年费政策：标准年费 200 元。", "火星移民贷款怎么办？")
    reply = _MOCK.chat(prompt)
    assert "补充" in reply or "细节" in reply
    assert "政策资料" not in reply


def test_mock_llm_answers_when_context_overlaps():
    prompt = _answer_prompt("信用卡年费政策：标准年费 200 元。", "信用卡年费多少钱？")
    reply = _MOCK.chat(prompt)
    assert "政策资料" in reply


def test_mock_llm_clarifies_when_context_empty():
    reply = _MOCK.chat(_answer_prompt("", "信用卡年费多少钱？"))
    assert "补充" in reply or "细节" in reply
