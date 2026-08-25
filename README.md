# banking-agent-hitl

**中文** | [English](README.en.md)

银行客服 Agent 原型：政策问答（RAG）、业务查询、工单创建、敏感操作人工审批。
基于 LangGraph 工作流编排，支持中断/恢复，全链路审计落库。

## 架构

```
用户 → CLI / FastAPI
         │
   LangGraph 工作流（SqliteSaver 持久化，支持中断恢复）
   classify_intent ─┬─ policy_qa（RAG 三态：回答/追问/拒答）
                    ├─ prepare_tool → check_permission ─┬─ 拒绝/缺参 → 结束
                    │                                   ├─ 只读 → execute_tool
                    │                                   └─ 敏感 → interrupt() 人工审批 → 执行/驳回
                    └─ chitchat
         │
   SQLite 审计（对话、检索命中、工具链路、审批记录）
```

安全设计核心：**权限裁决与审批闸门是纯框架代码，不依赖 LLM 输出**——
即使 prompt 被注入"免审批"指令，敏感工具也无法绕过 `interrupt()` 审批环节。

## 目录结构

```
├── configs/config.yaml          # 全部配置（LLM/Embedding/RAG阈值/存储），环境变量可覆盖
├── data/knowledge_base/         # 知识库文档（md/txt/pdf）
├── src/banking_agent/
│   ├── config.py                # pydantic-settings 配置加载
│   ├── llm/                     # LLM 层：openai_compat（OpenAI/vLLM/Ollama 通用）+ mock
│   ├── rag/                     # Embedding / FAISS 索引 / 三态检索器 / 文档加载(含PDF)
│   ├── auth/permissions.py      # 角色 + 工具授权 + 强制审批裁决
│   ├── tools/                   # 声明式工具注册（风险等级/参数schema/授权钩子）
│   ├── mcp/                     # MCP 工具链路：server(FastMCP) / client(同步桥接) / provider(信任侧注册)
│   ├── graph/                   # LangGraph 状态、节点、图编排、AgentService
│   ├── storage/                 # SQLite 审计表 + AuditLogger
│   ├── api/app.py               # FastAPI 骨架（/chat、/approvals）
│   ├── bootstrap.py             # 装配入口
│   └── cli.py                   # CLI demo
└── tests/                       # 政策问答 / 工具调用 / 安全注入 / MCP 工具链路 四类测试
```

## 快速开始

```bash
python3 -m venv venv-banking-agent
venv-banking-agent/bin/pip install -e . && venv-banking-agent/bin/pip install sentence-transformers pytest

# 预置演示对话（含 RAG 三态、越权拦截、转账审批）
venv-banking-agent/bin/python -m banking_agent.cli --scripted

# 交互模式
venv-banking-agent/bin/python -m banking_agent.cli

# 使用 Ollama 本地推理（需 ollama pull qwen3.5:9b）
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.ollama.yaml --scripted

# MCP 工具链路演示（银行工具经 MCP server 提供，stdio 自动拉起子进程）
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.mcp.yaml --scripted

# API 服务
venv-banking-agent/bin/uvicorn banking_agent.api.app:app --reload

# 测试（离线，Mock LLM + Mock Embedding）
venv-banking-agent/bin/pytest
```

版本记录见 [CHANGELOG.md](CHANGELOG.md)。

## 切换 LLM 来源（不改代码）

编辑 `configs/config.yaml`、指定 `--config` 配置文件（如 `configs/config.ollama.yaml`），或用环境变量：

```bash
# vLLM
export BANKING_AGENT_LLM__PROVIDER=openai_compat
export BANKING_AGENT_LLM__BASE_URL=http://localhost:8000/v1
export BANKING_AGENT_LLM__MODEL=Qwen/Qwen2.5-7B-Instruct

# Ollama
export BANKING_AGENT_LLM__BASE_URL=http://localhost:11434/v1
export BANKING_AGENT_LLM__MODEL=qwen2.5:7b

# OpenAI
export BANKING_AGENT_LLM__BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
```

## MCP 工具接入（不改图代码）

工具来源可配置切换（`tools.provider: local | mcp`）：`local` 为进程内直调；`mcp` 时银行工具由
MCP server（`src/banking_agent/mcp/server.py`）提供，agent 侧经 MCP 协议消费，LangGraph 图零改动。

```bash
# stdio（默认）：agent 自动拉起 server 子进程，无需额外部署
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.mcp.yaml --scripted

# streamable HTTP：工具服务独立部署
venv-banking-agent/bin/python -m banking_agent.mcp.server --http     # 终端 1 启动 server
# 将 configs/config.mcp.yaml 的 mcp.transport 改为 http 后再运行 agent  # 终端 2
```

信任边界设计：

- **服务器声明能力，客户端声明信任**：风险等级、最低角色、授权钩子取自 agent 侧本地策略——
  MCP 注解（`readOnlyHint` 等）按规范只是提示，不作为安全边界。注册集合 = 服务器能力 ∩ 本地
  信任声明；服务器多出的未声明工具默认不注册，本地声明而服务器缺失则启动即报错。
- 权限裁决与 `interrupt()` 审批闸门位置不变，仍在 agent 框架层——工具搬到进程外，闸门不动。
- 用户身份参数（如工单的 `user_id`）由框架注入，不出现在 LLM 可见的参数 schema。
- 新增一个工具 = server 加一个 `@server.tool` 函数 + agent 侧声明一份含安全元数据的 `ToolSpec`。

## RAG 三态判定

| 条件 | 行为 |
|---|---|
| top 相似度 ≥ `high_threshold` | 基于引用回答 |
| 介于两阈值之间 | 低置信，生成追问 |
| < `low_threshold` 或无命中 | 拒答（防幻觉） |

阈值在 `configs/config.yaml` 的 `rag` 段配置，需按所用 embedding 模型标定。

## 审批流（中断/恢复）

```python
reply = service.chat(thread_id, user, "向账户 ACC-002 转账 500 元")
# → {"status": "pending_approval", "approval_request": {...}}
# 图状态已持久化到 checkpoints.db，进程重启也能恢复

reply = service.resolve_approval(thread_id, approved=True, approver="supervisor")
# → {"status": "completed", "response": "操作已完成..."}
```
