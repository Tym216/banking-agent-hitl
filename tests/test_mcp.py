"""MCP 工具链路:能力∩信任注册、跨进程执行、权限与审批仍在 agent 侧生效。

test_mcp_end_to_end_flow 会经 stdio 拉起真实的 MCP server 子进程,
账本状态在服务端进程内——断言只能走对话黑盒,不能读本进程的 mock_bank。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest
from mcp import types

from banking_agent.bootstrap import DEMO_USERS, create_service
from banking_agent.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    MCPConfig,
    RAGConfig,
    StorageConfig,
    ToolsConfig,
)
from banking_agent.mcp import MCPToolClient, create_mcp_tool_registry


@pytest.fixture()
def mcp_service(tmp_path):
    cfg = AppConfig(
        llm=LLMConfig(provider="mock"),
        embedding=EmbeddingConfig(provider="mock", dim=512),
        rag=RAGConfig(
            kb_dir="tests/data/mock_kb",
            index_dir=str(tmp_path / "index"), high_threshold=0.22, low_threshold=0.13
        ),
        tools=ToolsConfig(provider="mcp"),
        mcp=MCPConfig(transport="stdio"),
        storage=StorageConfig(
            db_path=str(tmp_path / "audit.db"),
            checkpoint_db_path=str(tmp_path / "checkpoints.db"),
        ),
    )
    return create_service(config=cfg, rebuild_index=True)


def test_mcp_end_to_end_flow(mcp_service):
    """越权拦截(agent 侧 authorize)→ 本人查询 → 转账审批 → 服务端账本生效。"""
    alice = DEMO_USERS["u_alice"]
    thread_id = "t-mcp-1"

    # 越权:裁决发生在 agent 侧,请求根本不应到达 MCP server
    reply = mcp_service.chat(thread_id, alice, "帮我查一下账户 ACC-003 的余额")
    assert "越权" in reply["response"] or "拒绝" in reply["response"]
    assert "990000" not in reply["response"]

    reply = mcp_service.chat(thread_id, alice, "查一下我的余额")
    assert "58200.5" in reply["response"]

    # 敏感操作仍走 interrupt 审批;批准后账本(在 server 进程内)真实变更
    reply = mcp_service.chat(thread_id, alice, "向账户 ACC-002 转账 300 元")
    assert reply["status"] == "pending_approval"
    reply = mcp_service.resolve_approval(thread_id, True, DEMO_USERS["u_staff"], "已核实")
    assert reply["status"] == "completed"

    reply = mcp_service.chat(thread_id, alice, "查一下我的余额")
    assert "57900.5" in reply["response"]  # 58200.5 - 300


def test_client_raises_on_server_side_error():
    """服务端参数校验失败 → isError → 客户端抛异常(execute_tool 会转为可读回复)。"""
    client = MCPToolClient(MCPConfig(transport="stdio"))
    try:
        with pytest.raises(RuntimeError):
            client.call_tool("get_account_balance", {})  # 缺少必填参数
        result = client.call_tool("get_account_balance", {"account_id": "ACC-002"})
        assert result == {
            "ok": True, "account_id": "ACC-002", "balance": 1200.0, "currency": "CNY",
        }
    finally:
        client.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"MCP server 端口 {port} 未在 {timeout}s 内就绪")


def _start_http_server(port: int, token: str | None) -> subprocess.Popen:
    env = dict(os.environ)
    if token:
        env["MCP_AUTH_TOKEN"] = token
    return subprocess.Popen(
        [sys.executable, "-m", "banking_agent.mcp.server", "--http", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def test_http_transport():
    """streamable HTTP 传输 + Bearer token 鉴权:token 正确可连接并调用工具。"""
    port = _free_port()
    token = "test-secret-token"
    proc = _start_http_server(port, token)
    try:
        _wait_port(port, timeout=20)
        os.environ["MCP_AUTH_TOKEN"] = token
        client = None
        for _ in range(3):  # 端口可连到 ASGI 就绪之间可能有极短间隙,重试兜底
            try:
                client = MCPToolClient(
                    MCPConfig(
                        transport="http",
                        url=f"http://127.0.0.1:{port}/mcp",
                        timeout_s=15,
                    )
                )
                break
            except (TimeoutError, RuntimeError):
                time.sleep(0.5)
        assert client is not None, "HTTP MCP server 连接失败"
        try:
            names = {t.name for t in client.list_tools()}
            assert names == {"get_account_balance", "create_ticket", "submit_transaction"}
            result = client.call_tool("get_account_balance", {"account_id": "ACC-002"})
            assert result["balance"] == 1200.0
        finally:
            client.close()
            os.environ.pop("MCP_AUTH_TOKEN", None)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_http_transport_rejects_wrong_token():
    """错 token:server 返回 401,客户端连接失败。"""
    port = _free_port()
    proc = _start_http_server(port, "correct-token")
    try:
        _wait_port(port, timeout=20)
        os.environ["MCP_AUTH_TOKEN"] = "wrong-token"
        try:
            with pytest.raises(RuntimeError):
                MCPToolClient(MCPConfig(transport="http", url=f"http://127.0.0.1:{port}/mcp", timeout_s=10))
        finally:
            os.environ.pop("MCP_AUTH_TOKEN", None)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class _StubClient:
    """只实现 list_tools 的假客户端,用于测试注册的信任交集逻辑。"""

    def __init__(self, tool_names: list[str]) -> None:
        self._names = tool_names

    def list_tools(self):
        return [
            types.Tool(name=n, inputSchema={"type": "object"}) for n in self._names
        ]


def test_unknown_server_tool_is_not_registered():
    """服务器多报的工具不在本地信任策略中 → 默认拒绝,不注册。"""
    stub = _StubClient(
        ["get_account_balance", "create_ticket", "submit_transaction", "delete_all_accounts"]
    )
    registry = create_mcp_tool_registry(stub)
    names = {s.name for s in registry.list_specs()}
    assert names == {"get_account_balance", "create_ticket", "submit_transaction"}


def test_missing_declared_tool_fails_fast():
    """本地信任策略声明的工具服务器没提供 → 启动即报错,不带病运行。"""
    stub = _StubClient(["get_account_balance", "create_ticket"])
    with pytest.raises(RuntimeError, match="submit_transaction"):
        create_mcp_tool_registry(stub)
