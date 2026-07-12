"""规则驱动的 Mock LLM：离线、确定性，用于 demo 与自动化测试。

节点 prompt 中带有 [TASK:xxx] 标记，真实 LLM 忽略它，MockLLM 据此分派规则。
"""

from __future__ import annotations

import json
import re

from banking_agent.llm.base import Message

_INTENT_RULES: list[tuple[str, list[str]]] = [
    ("transfer", ["转账", "汇款", "转钱", "transfer"]),
    ("create_ticket", ["工单", "投诉", "报障", "反馈问题"]),
    ("balance_query", ["余额", "流水", "交易记录", "balance"]),
    ("policy_qa", ["政策", "费率", "利率", "分期", "信用卡", "年费", "挂失", "手续费", "额度", "还款", "贷款", "宽限期", "积分", "里程", "兑换", "限额"]),
]

_ACCOUNT_RE = re.compile(r"ACC-\d+")
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*元")


class MockLLMClient:
    def chat(self, messages: list[Message], **kwargs: object) -> str:
        prompt = "\n".join(m["content"] for m in messages)
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if "[TASK:classify_intent]" in prompt:
            return self._classify(user_text)
        if "[TASK:extract_params]" in prompt:
            return self._extract(prompt, user_text)
        if "[TASK:answer_with_context]" in prompt:
            return self._answer(prompt)
        if "[TASK:clarify]" in prompt:
            return "抱歉，我不完全确定您的问题。能否补充更多细节，例如具体的业务类型或卡种？"
        return "您好，我是银行客服助手，可以帮您查询政策、账户余额，或创建工单。"

    def _classify(self, text: str) -> str:
        for intent, keywords in _INTENT_RULES:
            if any(k in text for k in keywords):
                return intent
        return "chitchat"

    def _extract(self, prompt: str, text: str) -> str:
        accounts = _ACCOUNT_RE.findall(text)
        amount_m = _AMOUNT_RE.search(text)
        amount = float(amount_m.group(1).replace(",", "")) if amount_m else None
        if "tool=submit_transaction" in prompt:
            return json.dumps(
                {"to_account": accounts[0] if accounts else None, "amount": amount},
                ensure_ascii=False,
            )
        if "tool=get_account_balance" in prompt:
            return json.dumps(
                {"account_id": accounts[0] if accounts else None}, ensure_ascii=False
            )
        if "tool=create_ticket" in prompt:
            category = "complaint" if "投诉" in text else "general"
            return json.dumps(
                {"category": category, "summary": text[:200]}, ensure_ascii=False
            )
        return "{}"

    def _answer(self, prompt: str) -> str:
        ctx_m = re.search(r"<context>(.*?)</context>", prompt, re.S)
        ctx = ctx_m.group(1).strip() if ctx_m else ""
        first_para = ctx.split("\n\n")[0].strip()
        return f"根据我行政策资料：{first_para}\n（以上答案来自知识库检索结果）"
