"""prompt_ctx：身份/时间片段生成与标签渲染。"""

from __future__ import annotations

import re

from banking_agent.graph.prompt_ctx import (
    now_ctx,
    render_prompt_env,
    user_ctx,
)

_USER_CUSTOMER = {"user_id": "u_alice", "name": "Alice（客户）",
                  "role": "customer", "account_id": "ACC-001"}
_USER_STAFF = {"user_id": "u_staff", "name": "柜员王", "role": "staff",
               "account_id": None}


def test_user_ctx_minimal():
    s = user_ctx(_USER_CUSTOMER, "minimal")
    assert "客户" in s and "customer" in s
    assert "ACC-001" not in s  # minimal 不带账户


def test_user_ctx_full():
    s = user_ctx(_USER_CUSTOMER, "full")
    assert "Alice（客户）" in s
    assert "u_alice" in s
    assert "ACC-001" in s


def test_user_ctx_staff_without_account():
    s = user_ctx(_USER_STAFF, "full")
    assert "无绑定账户" in s


def test_user_ctx_none_and_unknown_level():
    assert user_ctx(None, "full") == ""
    assert "客户" in user_ctx(_USER_CUSTOMER, "whatever")  # 未知 level 回退 minimal


def test_now_ctx_format():
    s = now_ctx()
    assert re.match(r"^今天是 \d{4}年\d{1,2}月\d{1,2}日 周. \d{2}:\d{2}", s)
    assert "本地时区" in s


def test_render_replaces_present_tags():
    prompt = "请回答。\n<user_min/>\n<datetime/>"
    out = render_prompt_env(prompt, _USER_CUSTOMER)
    assert "<user_min/>" not in out and "<datetime/>" not in out
    assert "当前登录用户角色" in out
    assert "今天是 " in out


def test_render_missing_user_downgrades_tags():
    out = render_prompt_env("头\n<user_full/>\n<datetime/>", None)
    assert out == "头\n\n<datetime/>".replace("<datetime/>", now_ctx())
    # user 标签降级为空行，datetime 仍注入


def test_render_no_tags_unchanged():
    prompt = "没有任何标签的 prompt [TASK:x]"
    assert render_prompt_env(prompt, _USER_CUSTOMER) == prompt
