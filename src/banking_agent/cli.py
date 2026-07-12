"""交互式 CLI demo。

用法：
    python -m banking_agent.cli                # 交互模式
    python -m banking_agent.cli --scripted     # 跑一段预置演示对话

审批在终端内模拟：出现 pending_approval 时提示 y/n。
"""

from __future__ import annotations

import argparse
import json
import uuid

from banking_agent.bootstrap import DEMO_USERS, create_service
from banking_agent.graph.workflow import AgentService


def _handle_reply(service: AgentService, thread_id: str, reply: dict) -> None:
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
            approver="cli_supervisor",
            reason="CLI 手工审批",
        )
    print(f"\n🤖 {reply['response']}\n")


def interactive(config_path: str | None = None) -> None:
    service = create_service(config_path)
    print("=== 银行客服 Agent Demo ===")
    print("可选用户:", ", ".join(f"{k}({u.name})" for k, u in DEMO_USERS.items()))
    user_id = input("以哪个用户登录? [u_alice] > ").strip() or "u_alice"
    user = DEMO_USERS.get(user_id, DEMO_USERS["u_alice"])
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    print(f"已登录: {user.name}，会话: {thread_id}。输入 q 退出。\n")
    print("试试: 「信用卡年费怎么收？」「查一下我的余额」「向账户 ACC-002 转账 500 元」\n")

    while True:
        text = input(f"{user.name} > ").strip()
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break
        _handle_reply(service, thread_id, service.chat(thread_id, user, text))


def scripted(config_path: str | None = None) -> None:
    """预置演示：政策问答（三态）→ 余额查询 → 越权查询 → 转账审批。"""
    service = create_service(config_path)
    alice = DEMO_USERS["u_alice"]
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
            reply = service.resolve_approval(thread_id, True, "demo_supervisor", "演示批准")
        print(f"🤖 {reply['response']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted", action="store_true", help="运行预置演示对话")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 configs/config.yaml")
    args = parser.parse_args()
    scripted(args.config) if args.scripted else interactive(args.config)
