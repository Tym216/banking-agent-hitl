"""交互式 CLI demo。

用法：
    python -m banking_agent.cli                # 交互模式（用户名密码登录）
    python -m banking_agent.cli --scripted     # 跑一段预置演示对话

多会话与审批：
    - 客户发起敏感操作先出"确认草稿"，回复确认/取消/直接说修改内容
    - 确认态可随时切换其他话题（草稿挂起），"继续刚才的转账"可恢复
    - 客户确认后的审批不弹给客户，提示等待；下一条消息自动开新会话继续对话
    - staff（可审批）登录后可用 /approvals /approve 处理全局待审批
    - /threads /switch /status 管理会话与查看审批结果
    - 终端交互中输 / 后按 Tab 可补全命令
"""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Callable

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
    "zhou": "zhou123",
    "liang": "liang123",
    "zheng": "zheng123",
    "staff_approver": "staff123",
    "staff_viewer": "staff123",
}

_HELP = """可用命令：
  /threads                列出我的会话及审批状态
  /switch <序号|thread>   切到指定会话并查看最近消息
  /status                 查看当前会话的审批状态
  /approvals              列出全部待审批（仅可审批 staff）
  /approve <序号|thread> y|n [原因]   批准/拒绝（仅可审批 staff）
  /help                   显示帮助
  q / quit                退出"""


def _login(users: UserStore) -> User:
    print("=== 银行客服 Agent Demo ===")
    print("demo 账号: alice/bob/carol/zhou/liang/zheng (客户) | "
          "staff_approver (可审批) | staff_viewer (不可审批)")
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


class Session:
    """单用户交互会话：多会话管理 + 斜杠命令 + 审批角色分叉。"""

    def __init__(self, service: AgentService, user: User,
                 get_input: Callable[[str], str] = input) -> None:
        self.service = service
        self.user = user
        self.get_input = get_input
        self.thread_id = uuid.uuid4().hex
        self.blocked = False  # 当前会话有未决审批（客户侧：下条消息自动换新会话）

    # ── 主循环 ──────────────────────────────────────

    def run(self) -> None:
        self.print(f"已登录: {self.user.name}（{self.user.role.value}），当前会话: {self.thread_id}")
        self.print("输入 /help 查看命令，q 退出。")
        while True:
            text = self.get_input(f"{self.user.name} > ").strip()
            if not text:
                continue
            if text.lower() in {"q", "quit", "exit"}:
                break
            if text.startswith("/"):
                self._command(text)
                continue
            # 客户当前会话被审批阻塞 → 惰性自动开新会话继续对话
            if self.blocked:
                self.thread_id = self._new_thread_id()
                self.blocked = False
            self._chat(text)

    def print(self, *args) -> None:
        print(*args)

    def _new_thread_id(self) -> str:
        tid = uuid.uuid4().hex
        self.print(f"（原会话等待审批，已开启新会话 {tid}）")
        return tid

    # ── 对话与审批回复 ───────────────────────────────

    def _chat(self, text: str) -> None:
        reply = self.service.chat(self.thread_id, self.user, text)
        self._handle_reply(reply)

    def _handle_reply(self, reply: dict) -> None:
        while reply["status"] == "pending_approval":
            if self.user.can_approve:
                reply = self._inline_approve(reply)
            else:
                self._print_pending_notice(reply)
                self.blocked = True
                return
            if reply["status"] == "error":
                self.print(f"⚠️  {reply['message']}")
                return
        self.print(f"\n🤖 {reply['response']}\n")

    def _print_pending_notice(self, reply: dict) -> None:
        req = reply["approval_request"]
        self.print(
            f"\n⚠️  已提交审批（{req['tool_name']}），本会话暂停，请等待工作人员处理。"
            f"可输入 /status 查看状态、/switch 回看结果。"
        )

    def _inline_approve(self, reply: dict) -> dict:
        """staff 内联审批：直接在当前会话确认 y/n。"""
        req = reply["approval_request"]
        self.print("\n⚠️  [审批请求] 敏感操作等待人工审批：")
        self.print(f"    工具: {req['tool_name']}")
        self.print(f"    参数: {json.dumps(req['tool_args'], ensure_ascii=False)}")
        self.print(f"    发起人: {req['requested_by']['name']}")
        answer = self.get_input("    批准该操作? [y/n] > ").strip().lower()
        return self.service.resolve_approval(
            self.thread_id,
            approved=answer == "y",
            approver=self.user,
            reason="CLI 手工审批",
        )

    # ── 斜杠命令 ────────────────────────────────────

    def _command(self, text: str) -> None:
        tokens = text.split()
        cmd, args = tokens[0].lower(), tokens[1:]
        if cmd == "/help":
            self.print(_HELP)
        elif cmd == "/threads":
            self._cmd_threads()
        elif cmd == "/switch":
            self._cmd_switch(args[0] if args else "")
        elif cmd == "/status":
            self._cmd_status()
        elif cmd == "/approvals":
            self._cmd_approvals()
        elif cmd == "/approve":
            self._cmd_approve(args)
        else:
            self.print(f"未知命令 {cmd}，输入 /help 查看帮助")

    def _cmd_threads(self) -> None:
        result = self.service.list_threads(self.user)
        rows = result.get("threads", [])
        if not rows:
            self.print("（暂无会话）")
            return
        self.print(f"我的会话（最近 {len(rows)} 条，/switch <序号> 切换）：")
        for i, r in enumerate(rows, 1):
            mark = " [待审批]" if r["pending_count"] else ""
            title = (r.get("first_msg") or "(空会话)").replace("\n", " ")[:30]
            self.print(f"  {i}. {title}{mark}")

    def _cmd_switch(self, target: str) -> None:
        result = self.service.list_threads(self.user)
        rows = result.get("threads", [])
        tid = self._resolve_thread(rows, target)
        if tid is None:
            self.print(f"找不到会话 {target or '(空)'}（/threads 查看序号）")
            return
        self.thread_id = tid
        self.print(f"已切换到会话 {tid}，最近消息：")
        for m in self.service._audit.get_recent_messages(tid, limit=10):
            who = "客户" if m["role"] == "user" else "客服"
            self.print(f"  [{who}] {m['content'][:120]}")
        st = self.service.thread_status(tid, self.user)
        has_pending = any(a["status"] == "pending" for a in st.get("approvals", []))
        self.blocked = has_pending and not self.user.can_approve

    @staticmethod
    def _resolve_thread(rows: list[dict], target: str) -> str | None:
        if target.isdigit():
            i = int(target) - 1
            if 0 <= i < len(rows):
                return rows[i]["thread_id"]
            return None
        return target if any(r["thread_id"] == target for r in rows) else None

    def _cmd_status(self) -> None:
        st = self.service.thread_status(self.thread_id, self.user)
        if st.get("status") == "error":
            self.print(f"⚠️  {st['message']}")
            return
        approvals = st.get("approvals", [])
        if not approvals:
            self.print("（本会话暂无审批记录）")
            return
        label = {"pending": "待审批", "approved": "已批准", "rejected": "已拒绝"}
        for a in approvals:
            approver = a.get("approver") or "-"
            reason = f" 意见: {a['reason']}" if a.get("reason") else ""
            self.print(
                f"  [{label.get(a['status'], a['status'])}] {a['requested_at']}"
                f"  审批人: {approver}{reason}"
            )

    def _cmd_approvals(self) -> None:
        result = self.service.list_pending_approvals(self.user)
        if result.get("status") == "error":
            self.print(f"⚠️  {result['message']}")
            return
        rows = result.get("approvals", [])
        if not rows:
            self.print("（暂无待审批）")
            return
        self.print("待审批列表：")
        for i, a in enumerate(rows, 1):
            self.print(
                f"  {i}. thread={a['thread_id']} | {a['requester_name'] or a['user_id']}"
                f" | {a['tool_name']} {json.dumps(a['args'], ensure_ascii=False)}"
                f" | 请求于 {a['requested_at']}"
            )
        self.print("用 /approve <序号|thread> y|n [原因] 处理")

    def _cmd_approve(self, args: list[str]) -> None:
        if len(args) < 2:
            self.print("用法: /approve <序号|thread> y|n [原因]")
            return
        result = self.service.list_pending_approvals(self.user)
        if result.get("status") == "error":
            self.print(f"⚠️  {result['message']}")
            return
        rows = result.get("approvals", [])
        target, verb = args[0], args[1].lower()
        reason = " ".join(args[2:])
        tid = self._resolve_thread(rows, target) if rows else None
        if tid is None:
            self.print(f"找不到待审批 {target}（/approvals 查看序号）")
            return
        if verb not in ("y", "n", "yes", "no"):
            self.print("用法: /approve <序号|thread> y|n [原因]")
            return
        reply = self.service.resolve_approval(
            tid, approved=verb in ("y", "yes"), approver=self.user, reason=reason
        )
        if reply["status"] == "error":
            self.print(f"⚠️  {reply['message']}")
        else:
            self.print(f"✓ 已{'批准' if verb in ('y', 'yes') else '拒绝'}：{reply['response']}")


_COMMANDS = ["/help", "/threads", "/switch", "/status", "/approvals", "/approve"]


def _setup_readline_completion() -> None:
    """终端交互时注册 Tab 补全（/ 开头的命令）；非 tty/无 readline 时静默跳过。"""
    try:
        import readline
        import sys
    except ImportError:
        return
    if not sys.stdin.isatty():
        return
    readline.set_completer(
        lambda text, state: next(
            (c for i, c in enumerate(c for c in _COMMANDS if c.startswith(text))
             if i == state),
            None,
        )
    )
    readline.parse_and_bind("tab: complete")


def interactive(config_path: str | None = None) -> None:
    service = create_service(config_path)
    users = init_users(service._audit.conn)
    user = _login(users)
    _setup_readline_completion()
    Session(service, user).run()


def scripted(config_path: str | None = None) -> None:
    """预置演示：政策问答 → 余额查询 → 越权查询 → 转账审批。"""
    from banking_agent.bootstrap import DEMO_USERS

    service = create_service(config_path)
    alice = DEMO_USERS["u_alice"]
    approver = DEMO_USERS["u_staff"]
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"

    script = [
        "信用卡年费怎么收取？有没有减免政策？",
        "彩虹积分兑换航空里程的比例是多少？",
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
        elif "即将" in reply["response"]:
            # 客户确认草稿 → 演示自动确认
            print(f"🤖 {reply['response']}")
            print("    （演示：客户确认）")
            reply = service.chat(thread_id, alice, "确认")
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
