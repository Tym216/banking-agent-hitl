"""把 MCP server 的工具物化为 ToolSpec 注入 ToolRegistry(agent 信任侧)。

信任边界设计:
- 服务器声明「能力」(工具与参数 schema),客户端声明「信任」——风险等级、
  最低角色、所有权授权钩子、参数契约全部取自本地 spec,不采信服务器元数据
  (MCP 的 readOnlyHint 等注解按规范是 untrusted hints,不能当安全边界)。
- 注册集合 = 服务器能力 ∩ 本地信任声明:本地声明了但服务器没提供 → 启动即报错
  (配置错误尽早暴露);服务器多出的未声明工具 → 一律不注册(默认拒绝)。
- 身份注入:user_id 这类主体参数由框架从会话注入,不进入 LLM 的参数 schema,
  与「客户转账付款账户框架强制填充」是同一原则。
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel

from banking_agent.auth.permissions import User
from banking_agent.mcp.client import MCPToolClient
from banking_agent.tools import ToolRegistry, create_default_registry

# 工具名 → 需要框架注入的用户身份参数名(不出现在 LLM 可见的参数 schema 中)
_USER_INJECTED_PARAM = {"create_ticket": "user_id"}


def _make_mcp_handler(client: MCPToolClient, tool_name: str):
    inject_field = _USER_INJECTED_PARAM.get(tool_name)

    def handler(user: User, params: BaseModel) -> dict[str, Any]:
        args = params.model_dump()
        if inject_field:
            args[inject_field] = user.user_id
        return client.call_tool(tool_name, args)

    return handler


def create_mcp_tool_registry(client: MCPToolClient) -> ToolRegistry:
    served = {t.name for t in client.list_tools()}
    registry = ToolRegistry()
    for spec in create_default_registry().list_specs():
        if spec.name not in served:
            raise RuntimeError(
                f"MCP server 未提供本地信任策略中声明的工具: {spec.name}"
            )
        # 同一份 spec(含全部安全元数据),仅把执行方式换成远程 MCP 调用
        registry.register(
            dataclasses.replace(spec, handler=_make_mcp_handler(client, spec.name))
        )
    # served - 本地声明 的多余工具:未被信任,不注册(默认拒绝)
    return registry
