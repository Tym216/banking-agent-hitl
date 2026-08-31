"""银行工具 MCP server:相当于「银行核心系统适配器」,只负责执行。

安全属性(风险等级 / 最低角色 / 所有权授权)不在这里声明——那些属于 agent 侧
的信任策略(见 mcp/provider.py)。本服务器通过 annotations 声明的 readOnlyHint
等,按 MCP 规范只是提示(untrusted hints),不构成安全边界。

启动:
    python -m banking_agent.mcp.server           # stdio(由 agent 作为子进程拉起)
    python -m banking_agent.mcp.server --http    # streamable HTTP,默认 127.0.0.1:8000/mcp
    python -m banking_agent.mcp.server --http --port 9000   # 自定义监听地址/端口

信任模型:
    stdio 模式下信任来自 OS 管道(只有拉起子进程的 agent 能对话),无需鉴权。
    HTTP 模式下 server 是独立网络服务,任何可达端口的进程都能调用工具——
    因此 HTTP 模式强制 Bearer token 鉴权:设置环境变量 MCP_AUTH_TOKEN 后,
    所有请求必须带 Authorization: Bearer <token>,否则 401。
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from banking_agent.tools import mock_bank

server = FastMCP("banking-tools", log_level="WARNING")  # stdio 模式下 INFO 日志会刷屏


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_account_balance(account_id: str) -> dict[str, Any]:
    """查询账户余额"""
    return mock_bank.get_balance(account_id)


@server.tool()
def create_ticket(user_id: str, category: str, summary: str) -> dict[str, Any]:
    """创建客服工单(投诉、挂失、报障等)。user_id 由 agent 框架注入,不来自 LLM。"""
    return mock_bank.create_ticket(user_id, category, summary)


@server.tool(annotations=ToolAnnotations(destructiveHint=True))
def submit_transaction(from_account: str, to_account: str, amount: float) -> dict[str, Any]:
    """提交转账交易"""
    return mock_bank.submit_transaction(from_account, to_account, amount)


class _TokenAuthMiddleware(BaseHTTPMiddleware):
    """静态 Bearer token 校验:HTTP 模式的最小鉴权。"""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("Authorization") != f"Bearer {self._token}":
            return JSONResponse(
                {"error": "unauthorized", "message": "缺少或无效的 MCP_AUTH_TOKEN"},
                status_code=401,
            )
        return await call_next(request)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="以 streamable HTTP 方式启动")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 模式监听地址")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 模式监听端口")
    args = parser.parse_args()
    if args.http:
        app = server.streamable_http_app()
        token = os.environ.get("MCP_AUTH_TOKEN")
        if not token:
            print("警告: 未设置 MCP_AUTH_TOKEN,HTTP 模式将拒绝所有请求", flush=True)
        else:
            app = _TokenAuthMiddleware(app, token)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    else:
        server.run(transport="stdio")
