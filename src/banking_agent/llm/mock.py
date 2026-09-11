"""规则驱动的 Mock LLM：离线、确定性，用于 demo 与自动化测试。

节点 prompt 中带有 [TASK:xxx] 标记，真实 LLM 忽略它，MockLLM 据此分派规则。
"""

from __future__ import annotations

import json
import re

from langsmith import traceable

from banking_agent.llm.base import Message

_INTENT_RULES: list[tuple[str, list[str]]] = [
    ("transfer", ["转账", "汇款", "转钱", "transfer"]),
    ("create_ticket", ["工单", "投诉", "报障", "反馈问题"]),
    ("balance_query", ["余额", "流水", "交易记录", "balance"]),
    ("policy_qa", ["政策", "费率", "利率", "分期", "信用卡", "年费", "挂失", "手续费", "额度", "还款", "贷款", "宽限期", "积分", "里程", "兑换", "限额"]),
]

_ACCOUNT_RE = re.compile(r"ACC-\d+")
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*元")
# 与 mock_bank.lookup_account 的别名注册表一致（测试镜像，真实 LLM 自行泛化）。
# ASCII 字母数字边界而非 \b：CJK 邻接也可命中（"我是bob本人"→bob），
# 但仍不命中 bobby/abob 这类前后缀。
_NAME_ACCOUNT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:alice|bob|carol)(?![A-Za-z0-9_])", re.I)
_FROM_RE = re.compile(r"从\s*(ACC-\d+|alice|bob|carol)", re.I)
_TO_RE = re.compile(
    r"(?:转给|转到|转至|汇给|汇至|给)\s*(ACC-\d+|alice|bob|carol)", re.I
)


def _account_terms(text: str) -> list[str]:
    """按出现顺序收集账号/姓名候选（大小写不敏感）。"""
    found: list[str] = []
    for m in _ACCOUNT_RE.finditer(text):
        found.append(m.group(0))
    for m in _NAME_ACCOUNT_RE.finditer(text):
        found.append(m.group(0))
    # 合并为出现顺序（按原文本 index 排序去重）
    tokens = [
        (m.start(), m.group(0))
        for m in _ACCOUNT_RE.finditer(text)
    ] + [(m.start(), m.group(0)) for m in _NAME_ACCOUNT_RE.finditer(text)]
    tokens.sort()
    seen: list[str] = []
    for _, tok in tokens:
        if tok.lower() not in [s.lower() for s in seen]:
            seen.append(tok)
    return seen


class MockLLMClient:

    @traceable(run_type="llm")
    def chat(self, messages: list[Message], **kwargs: object) -> str:
        prompt = "\n".join(m["content"] for m in messages)
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if "[TASK:classify_intent]" in prompt:
            intent = self._classify(user_text)
            print(f"[DEBUG] classify_intent -> {intent}", flush=True)
            return intent
        if "[TASK:extract_params]" in prompt:
            return self._extract(prompt, user_text)
        if "[TASK:answer_with_context]" in prompt:
            return self._answer(prompt, user_text)
        if "[TASK:clarify]" in prompt:
            return "抱歉，我不完全确定您的问题。能否补充更多细节，例如具体的业务类型或卡种？"
        return "您好，我是银行客服助手，可以帮您查询政策、账户余额，或创建工单。"

    def _classify(self, text: str) -> str:
        for intent, keywords in _INTENT_RULES:
            if any(k in text for k in keywords):
                return intent
        return "chitchat"

    def _extract(self, prompt: str, text: str) -> str:
        amount_m = _AMOUNT_RE.search(text)
        amount = float(amount_m.group(1).replace(",", "")) if amount_m else None
        if "tool=submit_transaction" in prompt:
            terms = _account_terms(text)
            from_m = _FROM_RE.search(text)
            from_c = from_m.group(1) if from_m else None
            to_m = _TO_RE.search(text)
            if to_m:
                to_c = to_m.group(1)
            else:
                # 回退：首个非 from 的账号/姓名候选（兼容"向账户 ACC-002 转账"句式）
                to_c = next((t for t in terms if t.lower() != (from_c or "").lower()), None)
            return json.dumps(
                {"from_account": from_c, "to_account": to_c, "amount": amount},
                ensure_ascii=False,
            )
        if "tool=get_account_balance" in prompt:
            terms = _account_terms(text)
            account = terms[0] if terms else None
            if account is None:
                # 通用回退：捕获"<姓名>的余额"中的姓名（模拟真实 LLM 上报未知名字）；
                # 排除含代词（我/你/他/她/它）的误捕
                m = re.search(r"([\u4e00-\u9fff]{2,3})的余额", text)
                cand = m.group(1) if m else None
                if cand and not any(c in cand for c in "你我他她它"):
                    account = cand
            return json.dumps({"account_id": account}, ensure_ascii=False)
        if "tool=create_ticket" in prompt:
            category = None
            if "投诉" in text:
                category = "complaint"
            elif "挂失" in text or "盗刷" in text or "丢失" in text:
                category = "card_loss"
            elif "报障" in text or "反馈" in text:
                category = "general"
            # 修改/恢复类指令不含新摘要内容 → summary 置 null（保留草稿原摘要）
            if any(k in text for k in ("继续", "改成", "修改", "换成", "回到")):
                summary = None
            else:
                summary = text[:200]
            return json.dumps(
                {"category": category, "summary": summary}, ensure_ascii=False
            )
        return "{}"

    def _answer(self, prompt: str, user_text: str = "") -> str:
        ctx_m = re.search(r"<context>(.*?)</context>", prompt, re.S)
        ctx = ctx_m.group(1).strip() if ctx_m else ""
        first_para = ctx.split("\n\n")[0].strip()
        # 模拟"LLM 自主判断资料不足"：问题与上下文无任何 CJK bigram 重合 → 追问
        if first_para and user_text:
            q_bigrams = set(re.findall(r"[\u4e00-\u9fff]{2}", user_text))
            ctx_bigrams = set(re.findall(r"[\u4e00-\u9fff]{2}", ctx))
            if q_bigrams and not (q_bigrams & ctx_bigrams):
                return "抱歉，资料里似乎没有直接覆盖您问的内容，能否再补充一些细节？"
        if not first_para:
            return "抱歉，我不完全确定您的问题。能否补充更多细节，例如具体的业务类型或卡种？"
        return f"根据我行政策资料：{first_para}\n（以上答案来自知识库检索结果）"
