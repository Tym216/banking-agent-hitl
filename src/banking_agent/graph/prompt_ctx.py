"""Prompt 环境上下文注入：身份/时间由渲染器按标签替换，模板与值解耦。

模板内可放置三个标签（<user_min/> / <user_full/> / <datetime/>），
调用节点在最终 prompt 上执行 render_prompt_env 完成注入：
- user_min   仅角色（意图分类用）
- user_full  角色+姓名+user_id+账户（参数抽取用，消解"我的账户"歧义）
- datetime   本地时区中文完整时间（政策问答的时间有效性用）

注意：身份信息只作为 LLM 上下文，权限/越权裁决仍由框架层硬校验。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

_ROLE_LABEL = {"customer": "客户", "staff": "柜员", "admin": "管理员"}
_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

_TAG_MIN = "<user_min/>"
_TAG_FULL = "<user_full/>"
_TAG_TIME = "<datetime/>"


def user_ctx(state_user: dict[str, Any] | None, level: str) -> str:
    """生成当前登录用户身份片段。

    level:
      "minimal" — 只要角色：当前登录用户角色：客户（customer）
      "full"    — 角色+姓名+user_id+账户
    未知 level 回退 minimal；state_user 为空返回空串（不注入身份）。
    """
    if not state_user:
        return ""
    role = str(state_user.get("role", ""))
    label = _ROLE_LABEL.get(role, role)
    if level == "minimal":
        return f"当前登录用户角色：{label}（{role}）"
    if level == "full":
        name = state_user.get("name", "")
        uid = state_user.get("user_id", "")
        account = state_user.get("account_id") or "无绑定账户"
        return (
            f"当前登录用户：{name}（{label}）｜用户ID：{uid}"
            f"｜角色：{label}｜账户：{account}"
        )
    return f"当前登录用户角色：{label}（{role}）"


def now_ctx() -> str:
    """本地时区中文完整时间。"""
    now = datetime.now()
    wd = _WEEKDAYS[now.weekday()]
    return (
        f"今天是 {now.year}年{now.month}月{now.day}日 {wd} "
        f"{now.hour:02d}:{now.minute:02d}（系统本地时区）"
    )


def render_prompt_env(prompt: str, state_user: dict[str, Any] | None = None) -> str:
    """按模板内标签替换环境上下文；无标签或不可用时原样/降级返回。"""
    out = prompt
    out = out.replace(_TAG_MIN, user_ctx(state_user, "minimal"))
    out = out.replace(_TAG_FULL, user_ctx(state_user, "full"))
    out = out.replace(_TAG_TIME, now_ctx())
    return out
