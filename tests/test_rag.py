"""政策问答类：RAG 三态判定。"""

from banking_agent.bootstrap import DEMO_USERS


def test_high_confidence_policy_question_gets_answer(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-rag-1", alice, "信用卡年费政策是怎么收取的？")
    assert reply["status"] == "completed"
    assert "年费" in reply["response"]
    # 高置信回答必须带知识库来源，而不是模型裸答
    assert "知识库" in reply["response"] or "政策资料" in reply["response"]


def test_unknown_topic_is_refused(service):
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-rag-2", alice, "请问怎样申请办理火星移民贷款业务呀？")
    assert reply["status"] == "completed"
    assert "无法回答" in reply["response"] or "没有找到" in reply["response"]


def test_low_confidence_clarify_then_followup_answers(service):
    """低置信 → 追问 → 用户补充后合并原问题检索 → 高置信回答（多轮记忆）。"""
    alice = DEMO_USERS["u_alice"]
    reply = service.chat("t-rag-3", alice, "那年费呢")
    assert reply["status"] == "completed"
    assert "补充" in reply["response"] or "细节" in reply["response"]  # 追问语气

    reply = service.chat("t-rag-3", alice, "我想了解信用卡的年费收费标准和减免政策")
    assert reply["status"] == "completed"
    assert "年费" in reply["response"]
    assert "政策资料" in reply["response"] or "知识库" in reply["response"]
