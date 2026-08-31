"""交互式 CLI demo。

用法：
    python -m banking_agent.cli                # 交互模式（用户名密码登录）
    python -m banking_agent.cli --scripted     # 跑一段预置演示对话

审批在终端内模拟：出现 pending_approval 时提示 y/n，
需要以 staff（可审批）账号登录。
"""

from __future__ import annotations

import argparse
import json
import uuid

from banking_agent.auth.accounts import UserStore, init_users
from banking_agent.auth.permissions import User
from banking_agent.bootstrap import create_service
from banking_agent.config import load_config
from banking_agent.graph.workflow import AgentService
from banking_agent.storage import connect

_DEMO_ACCOUNTS = {
    "alice": "alice123",
    "bob": "bob123",
    "carol": "carol123",
    "staff_approver": "staff123",
    "staff_viewer": "staff123",
}


def _login(service: AgentService, users: UserStore) -> User:
    print("=== 银行客服 Agent Demo ===")
    print("demo 账号: alice/bob/carol (客户) | staff_approver (可审批) | staff_viewer (不可审批)")
    while True:
        username = input("用户名 > ").strip()
        password = input("密码 > ").strip()
        if not username or not password:
            continue
        user = users.authenticate(username, password)
        if user is None:
            print("用户名或密码错误，请重试。\n")
            continue
        return user


def _handle_reply(service: AgentService, thread_id: str, reply: dict, approver: User) -> None:
    while reply["status"] == "pending_approval":
        req = reply["approval_request"]
        print("\n⚠️  [审批请求] 敏感操作等待人工审批：")
        print(f"    工具: {req['tool_name']}")
        print(f"    参数: {json.dumps(req['tool_args'], ensure_ascii=False)}")
        print(f"    发起人: {req['requested_by']['name']}")
        answer = input("    批准该操作? [y/n] > ").strip().lower()
        reply = service.resolve_approval(
            thread_id,
            approved=answer == "y",
            approver=approver,
            reason="CLI 手工审批",
        )
        if reply["status"] == "error":
            print(f"\n⚠️  {reply['message']}")
            return
    print(f"\n🤖 {reply['response']}\n")


def interactive(config_path: str | None = None) -> None:
    service = create_service(config_path)
    users = init_users(service._audit.conn)
    user = _login(service, users)
    thread_id = uuid.uuid4().hex
    print(f"\n已登录: {user.name}（{user.role.value}），会话: {thread_id}。输入 q 退出。\n")
    print("试试: 「信用卡年费怎么收？」「查一下我的余额」「向账户 ACC-002 转账 500 元」\n")

    while True:
        text = input(f"{user.name} > ").strip()
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break
        _handle_reply(service, thread_id, service.chat(thread_id, user, text), user)


def scripted(config_path: str | None = None) -> None:
    """预置演示：政策问答（三态）→ 余额查询 → 越权查询 → 转账审批。"""
    from banking_agent.bootstrap import DEMO_USERS

    service = create_service(config_path)
    alice = DEMO_USERS["u_alice"]
    approver = DEMO_USERS["u_staff"]
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"

    script = [
        "信用卡年费怎么收取？有没有减免政策？",
        "彩虹积分兑换航空里程的比例是多少？",  # 知识库没有 → 应拒答
        "查一下我的账户余额",
        "帮我查账户 ACC-003 的余额",             # 越权 → 应拒绝
        "向账户 ACC-002 转账 500 元",            # 敏感 → 走审批
    ]
    for text in script:
        print(f"\n{'=' * 60}\n👤 {alice.name}: {text}")
        reply = service.chat(thread_id, alice, text)
        if reply["status"] == "pending_approval":
            req = reply["approval_request"]
            print(f"⚠️  审批请求: {req['tool_name']} {json.dumps(req['tool_args'], ensure_ascii=False)}")
            print("    （演示：审批人批准）")
            reply = service.resolve_approval(thread_id, True, approver, "演示批准")
        print(f"🤖 {reply['response']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted", action="store_true", help="运行预置演示对话")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 configs/config.yaml")
    args = parser.parse_args()
    scripted(args.config) if args.scripted else interactive(args.config)
