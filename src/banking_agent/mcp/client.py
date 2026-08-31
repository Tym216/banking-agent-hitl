"""MCP 客户端(同步外观,内部异步桥接)。

官方 mcp SDK 是 async-only。为了不把异步性扩散到图/服务层(它们保持同步),
本类把整个会话生命周期放进一个后台事件循环线程里的**单个任务**中
(stdio/HTTP 连接使用 anyio cancel scope,必须在同一任务内进出),同步调用
经 run_coroutine_threadsafe 桥接提交。

未来全栈 async 时:删掉本桥接,handler 直接 await session.call_tool 即可。
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

try:  # mcp>=1.10 改名;旧版回退到 streamablehttp_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover
    from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

from banking_agent.config import MCPConfig


class MCPToolClient:
    def __init__(self, cfg: MCPConfig) -> None:
        self._cfg = cfg
        self._session: ClientSession | None = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self._shutdown: asyncio.Event | None = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-client-loop", daemon=True
        )
        self._thread.start()
        self._lifecycle = asyncio.run_coroutine_threadsafe(self._run_session(), self._loop)
        if not self._ready.wait(timeout=cfg.timeout_s):
            raise TimeoutError("MCP server 连接超时")
        if self._startup_error is not None:
            raise RuntimeError(f"MCP server 连接失败: {self._startup_error}")
        atexit.register(self.close)

    async def _run_session(self) -> None:
        """会话全生命周期:连接 → 就绪 → 等待关闭信号 → 原任务内退出。"""
        self._shutdown = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                if self._cfg.transport == "stdio":
                    params = StdioServerParameters(
                        command=sys.executable, args=["-m", "banking_agent.mcp.server"]
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                else:  # http
                    token = os.environ.get(self._cfg.auth_token_env)
                    if token:
                        http_client = httpx.AsyncClient(
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=self._cfg.timeout_s,
                        )
                        await stack.enter_async_context(http_client)
                        read, write, _ = await stack.enter_async_context(
                            streamable_http_client(self._cfg.url, http_client=http_client)
                        )
                    else:
                        read, write, _ = await stack.enter_async_context(
                            streamable_http_client(self._cfg.url)
                        )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._shutdown.wait()
        except BaseException as e:
            self._startup_error = e
            self._ready.set()
        finally:
            self._session = None

    def _submit(self, coro) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=self._cfg.timeout_s
        )

    def list_tools(self) -> list[types.Tool]:
        assert self._session is not None
        return self._submit(self._session.list_tools()).tools

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MCP 会话不可用(未连接或已关闭)")
        result = self._submit(self._session.call_tool(name, arguments=args))
        return self._to_dict(result)

    @staticmethod
    def _to_dict(result: types.CallToolResult) -> dict[str, Any]:
        if result.isError:
            texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
            raise RuntimeError("; ".join(texts) or "MCP 工具调用失败")
        data = result.structuredContent
        if isinstance(data, dict):
            # FastMCP 对非对象返回值会包一层 {"result": ...},对象返回原样
            if set(data.keys()) == {"result"} and isinstance(data["result"], dict):
                return data["result"]
            return data
        for c in result.content:
            if isinstance(c, types.TextContent):
                try:
                    parsed = json.loads(c.text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
                return {"ok": True, "text": c.text}
        return {"ok": True}

    def close(self) -> None:
        if self._loop.is_closed():
            return
        if self._shutdown is not None and self._session is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)
            try:
                self._lifecycle.result(timeout=5)
            except Exception:
                pass  # 关闭失败不影响进程退出;子进程随管道关闭而结束
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()
